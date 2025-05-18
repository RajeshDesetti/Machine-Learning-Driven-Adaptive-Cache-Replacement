#include "ml.h" // Include the header for this implementation file

// Include standard libraries
#include <fstream>         // For std::ifstream
#include <vector>          // For std::vector
#include <string>          // For std::string
#include <map>             // For std::map
#include <cmath>           // For std::log1p
#include <stdexcept>       // For std::runtime_error, std::out_of_range
#include <limits>          // For std::numeric_limits
#include <iostream>        // For potential error logging (std::cerr)

// Include required ChampSim headers
#include "cache.h"         // Provides CACHE class definition and members like OFFSET_BITS, NUM_SET
#include "champsim.h"      // Provides champsim::lg2 and other utilities
#include "util/to_underlying.h" // Provides champsim::to_underlying

// Include JSON library
#include "nlohmann/json.hpp" // JSON parsing

// Alias for JSON library namespace
using json = nlohmann::json;

// --- Private Helper Implementation ---
int MLReplacementPolicy::get_feature_index(const std::string& name) const {
    auto it = feature_mapping.find(name);
    if (it == feature_mapping.end()) {
        // Provide more context in the error message
        std::string available_keys = "{ ";
        for(const auto& pair : feature_mapping) { available_keys += "'" + pair.first + "' "; }
        available_keys += "}";
        throw std::runtime_error("ML_POLICY ERROR: Feature '" + name + "' not found in mapping during lookup. Available keys: " + available_keys);
    }
    return it->second; // Return the index (int value) from the map
}

// --- Constructor Implementation ---
MLReplacementPolicy::MLReplacementPolicy(CACHE* cache)
    : champsim::modules::replacement(cache), // Initialize base class with CACHE pointer
      assoc(cache->NUM_WAY),                // Get associativity from the passed cache object
      env(ORT_LOGGING_LEVEL_WARNING, "MLReplacement"), // Initialize ONNX environment
      session(nullptr),                     // Initialize session explicitly to nullptr
      per_set_features(cache->NUM_SET) {    // Resize state vector based on cache sets

    // Load configuration file (model_config.json)
    const std::string config_path = "replacement/ml/model_config.json";
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("ML_POLICY ERROR: Failed to open ML config file: " + config_path);
    }

    json config;
    try {
        config = json::parse(config_file);
    } catch (const json::parse_error& e) {
        throw std::runtime_error("ML_POLICY ERROR: Failed to parse ML config file (" + config_path + "): " + e.what());
    }

    // Load preprocessing parameters and feature mapping from JSON
    try {
        scaler_mean = config.at("scaler_mean").get<std::vector<float>>();
        scaler_scale = config.at("scaler_scale").get<std::vector<float>>();
        label_classes = config.at("label_encoder_classes").get<std::vector<int>>();

        if (scaler_mean.empty() || scaler_mean.size() != scaler_scale.size()) {
            throw std::runtime_error("ML_POLICY ERROR: Scaler mean/scale size mismatch or empty in config.");
        }

        feature_mapping.clear();
        const auto& mapping = config.at("feature_mapping");
        for (const auto& [onnx_name, orig_name_json] : mapping.items()) {
            if (onnx_name.length() <= 1 || onnx_name[0] != 'f') {
                 throw std::runtime_error("ML_POLICY ERROR: Invalid ONNX feature name format in config: " + onnx_name);
            }
            std::string orig_name = orig_name_json.get<std::string>();
            try {
                int index = std::stoi(onnx_name.substr(1));
                if (index < 0 || static_cast<size_t>(index) >= scaler_mean.size()) {
                     throw std::out_of_range("Feature index " + std::to_string(index) + " from '" + onnx_name + "' is out of bounds for scaler size (" + std::to_string(scaler_mean.size()) + ").");
                }
                feature_mapping[orig_name] = index;
            } catch (const std::exception& e) {
                 throw std::runtime_error("ML_POLICY ERROR: Error parsing feature index from '" + onnx_name + "': " + e.what());
            }
        }

        if (feature_mapping.size() != scaler_mean.size()) {
             throw std::runtime_error("ML_POLICY ERROR: Final feature mapping size (" + std::to_string(feature_mapping.size()) +
                                      ") doesn't match scaler size (" + std::to_string(scaler_mean.size()) + ")");
        }

    } catch (const json::exception& e) {
        throw std::runtime_error("ML_POLICY ERROR: Error accessing elements in ML config JSON ('" + config_path + "'): " + std::string(e.what()));
    } catch (const std::exception& e) {
         throw std::runtime_error("ML_POLICY ERROR: Error loading preprocessing parameters: " + std::string(e.what()));
    }

    // Initialize ONNX Runtime session
    const std::string model_path = "replacement/ml/model.onnx";
    try {
        Ort::SessionOptions session_options;
        session = Ort::Session(env, model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ML_POLICY ERROR: Failed to initialize ONNX Runtime session (" + model_path + "): " + std::string(e.what()) + " (Code: " + std::to_string(e.GetOrtErrorCode()) + ")");
    }
    if (!session) {
        throw std::runtime_error("ML_POLICY ERROR: ONNX Session is invalid after initialization attempt for " + model_path);
    }
}

// --- Method Implementations ---

void MLReplacementPolicy::initialize_replacement() {
    for (auto& sf : per_set_features) {
        sf.last_address = 0;
        sf.last_pc = 0;
    }
      std::cout << "ML_POLICY INFO: MLReplacementPolicy initialized for cache " << intern_->NAME << std::endl;
}

long MLReplacementPolicy::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set,
                                     const CACHE::BLOCK* current_set, champsim::address ip,
                                     champsim::address full_addr, access_type type) {
    // 1. Basic Sanity Checks
    if (set < 0 || static_cast<std::size_t>(set) >= per_set_features.size()) {
        std::cerr << "ML_POLICY WARNING: Invalid set index " << set << " received in find_victim. Max sets: " << per_set_features.size() << ". Returning way 0." << std::endl;
        return 0;
    }
    if (feature_mapping.empty()) {
         throw std::runtime_error("ML_POLICY ERROR: Feature mapping is empty in find_victim. Check config loading.");
    }
     if (!session) {
         throw std::runtime_error("ML_POLICY ERROR: ONNX session is not valid in find_victim.");
     }

    const auto& sf = per_set_features.at(set);

    // 2. Prepare Feature Vector
    std::vector<float> features(feature_mapping.size());
    uint64_t current_address = full_addr.to<uint64_t>();
    uint64_t current_pc = ip.to<uint64_t>();

    // --- Correct tag calculation using champsim::to_underlying ---
    uint64_t offset_bits = champsim::to_underlying(intern_->OFFSET_BITS);
    uint64_t num_sets = intern_->NUM_SET;
    uint64_t set_bits = (num_sets <= 1) ? 0 : champsim::lg2(num_sets);
    uint64_t total_shift = offset_bits + set_bits;
    uint64_t tag = (total_shift < 64) ? (current_address >> total_shift) : 0;
    // --- End Fix ---

    uint8_t access_type_uint = static_cast<uint8_t>(type);
    uint32_t reuse_dist = 0; // Assumed 0 as it's not provided here by ChampSim

    try {
        // Populate features using helper function for safety and clarity
        features.at(get_feature_index("Address")) = static_cast<float>(current_address);
        features.at(get_feature_index("Set")) = static_cast<float>(set);
        features.at(get_feature_index("Tag")) = static_cast<float>(tag); // Use calculated tag
        features.at(get_feature_index("PC")) = static_cast<float>(current_pc);
        // features.at(get_feature_index("Cycle")) = static_cast<float>(intern_->current_cycle); // Example if needed
        features.at(get_feature_index("Type")) = static_cast<float>(access_type_uint);
        features.at(get_feature_index("ReuseDist")) = static_cast<float>(reuse_dist);

        features.at(get_feature_index("Address_diff")) = static_cast<float>(current_address - sf.last_address);
        features.at(get_feature_index("PC_diff")) = static_cast<float>(current_pc - sf.last_pc);
        if (feature_mapping.count("ReuseDist_log")) {
             features.at(get_feature_index("ReuseDist_log")) = std::log1p(static_cast<float>(reuse_dist));
        }

    } catch (const std::out_of_range& e) {
         throw std::runtime_error("ML_POLICY ERROR: Out of range error populating features vector: " + std::string(e.what()));
    } catch (const std::runtime_error& e) {
         throw std::runtime_error("ML_POLICY ERROR: Error getting feature index during population: " + std::string(e.what()));
    }

    // 3. Apply Scaling
    if (features.size() != scaler_mean.size()) {
         throw std::runtime_error("ML_POLICY ERROR: Internal error: Feature vector size mismatch before scaling.");
    }
    for (size_t i = 0; i < features.size(); ++i) {
        if (scaler_scale.at(i) == 0.0f) {
             features[i] = 0.0f;
        } else {
            features[i] = (features[i] - scaler_mean.at(i)) / scaler_scale.at(i);
        }
    }

    // 4. Prepare ONNX Input Tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(features.size())};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, features.data(), features.size(),
        input_shape.data(), input_shape.size());

    // 5. Run ONNX Inference
    const char* input_names[] = {"float_input"}; // Ensure this matches the actual input name in model.onnx
    // --- Use the correct output name "label" ---
    const char* output_names[] = {"label"};
    // --- End Fix ---
    std::vector<Ort::Value> output_tensors;
    try {
        // Request 1 input ("float_input") and 1 output ("label")
        output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    } catch (const Ort::Exception& e) {
        // Error handling remains the same, but the cause was likely the output_names mismatch
        throw std::runtime_error("ML_POLICY ERROR: ONNX Runtime inference failed for set " + std::to_string(set) + ": " + std::string(e.what()) + " (Code: " + std::to_string(e.GetOrtErrorCode()) + ")");
    }

    // 6. Process Output Tensor
    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        throw std::runtime_error("ML_POLICY ERROR: Invalid or empty output tensor (expected 'label') from ONNX model for set " + std::to_string(set));
    }

    // Assuming the "label" output is int64 type as expected
    int64_t* predictions = output_tensors[0].GetTensorMutableData<int64_t>();
    int64_t predicted_label = predictions[0]; // Get the first element of the label tensor

    long victim_way = static_cast<long>(predicted_label % assoc);
    if (victim_way < 0) {
        victim_way += assoc;
    }

    return victim_way;
}

void MLReplacementPolicy::update_replacement_state(uint32_t triggering_cpu, long set, long way,
                                                  champsim::address full_addr, champsim::address ip,
                                                  champsim::address victim_addr, access_type type,
                                                  bool hit) {
    // Check for valid set index
     if (set < 0 || static_cast<std::size_t>(set) >= per_set_features.size()) {
         std::cerr << "ML_POLICY WARNING: Invalid set index " << set << " received in update_replacement_state. Max sets: " << per_set_features.size() << ". State not updated." << std::endl;
         return;
    }

    // Update the history state for this set using .at() for bounds check
    auto& sf = per_set_features.at(set);
    sf.last_address = full_addr.to<uint64_t>(); // Store uint64_t representation
    sf.last_pc = ip.to<uint64_t>();          // Store uint64_t representation
}

#ifndef ML_REPLACEMENT_H
#define ML_REPLACEMENT_H

// Include necessary ChampSim headers first
#include "cache.h" // Provides CACHE, CACHE::BLOCK, access_type, champsim::address
#include "modules.h" // Provides champsim::modules::replacement base class

// Include external/standard library headers
#include <onnxruntime_cxx_api.h> // ONNX Runtime
#include <vector>
#include <map>
#include <string>
#include <cstdint> // Standard integer types (uint32_t, uint64_t)

// Forward declaration is usually not needed if cache.h is included, but doesn't hurt
class CACHE;

class MLReplacementPolicy : public champsim::modules::replacement {
private:
    // --- Private Members ---
    const uint32_t assoc; // Cache associativity
    Ort::Env env; // ONNX Runtime environment
    Ort::Session session{nullptr}; // ONNX Runtime session (initialized to null)

    // Preprocessing parameters loaded from JSON
    std::vector<float> scaler_mean;
    std::vector<float> scaler_scale;
    std::vector<int> label_classes; // Original target way values (loaded but not used currently)
    std::map<std::string, int> feature_mapping; // Maps feature name to its vector index (0, 1, ...)

    // Per-set state tracking for engineered features
    struct SetFeatures {
        uint64_t last_address = 0; // Address of the previous access to this set
        uint64_t last_pc = 0;      // PC of the previous access to this set
    };
    std::vector<SetFeatures> per_set_features; // State for each set

    // --- Private Helpers ---
    // Safely gets the index for a feature name from the map
    int get_feature_index(const std::string& name) const;

public:
    // --- Constructor ---
    // Takes a pointer to the CACHE it belongs to (passed to base class)
    explicit MLReplacementPolicy(CACHE* cache);

    // --- Replacement Policy Interface Functions ---
    // These functions implement the policy logic called by the framework adapter.
    // They DO NOT use 'override' as they don't override virtuals in the direct base.
    // Their signatures match what the adapter (replacement_module_model) expects.

    // Called once during initialization
    void initialize_replacement();

    // Called by the cache when a victim block needs to be selected for replacement
    long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set,
                     const CACHE::BLOCK* current_set, champsim::address ip,
                     champsim::address full_addr, access_type type);

    // Called by the cache after an access (hit or fill) to update internal state
    void update_replacement_state(uint32_t triggering_cpu, long set, long way,
                                  champsim::address full_addr, champsim::address ip,
                                  champsim::address victim_addr, access_type type,
                                  bool hit);
};

#endif // ML_REPLACEMENT_H

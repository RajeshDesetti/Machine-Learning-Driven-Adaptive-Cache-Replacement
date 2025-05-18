 Cache replacement policies play a crucial role in determining the performance of mod
ern processors by managing which cache lines are retained or evicted. Existing stan
dard static replacement policies, does not adapt to dynamic nature of memory access
 patterns, and the diversity of computer programs. This project explores the application
 of Machine Learning (ML) to improve cache performance by learning access patterns
 and predicting optimal eviction candidates.
 The proposed methodology implement an XGBoost-based replacement policy in the
 ChampSim simulator, leveraging features such as memory addresses, program counters
 (PCs), access types, reuse distances, and temporal locality. The model is trained on trace
 data collected from cache operations, achieving 90% accuracy in predicting eviction
 candidates.
 The performance of the XGBoost based replacement policy is evaluated against stan
dard replacement policies across multiple benchmarks from the SPEC CPU2017 suite.
 Key metrics such as Instructions Per Cycle (IPC), cache hit rates are analyzed to as
sess improvements. Results demonstrate that the ML-driven policy can reduce cache
 misses and improve cache performance particularly in workloads with irregular access
 patterns

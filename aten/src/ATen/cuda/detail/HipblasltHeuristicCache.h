// Included only from CUDABlas.cpp under USE_ROCM.
// Hipify rewrites this to ATen/hip/detail/HipblasltHeuristicCache.h.
//
// Remembers hipblasLtMatmulAlgoGetHeuristic() results for repeated GEMMs.
// The result is plain data (a solution index plus the workspace limit) and
// refers to no handle, stream, descriptor, or library memory. The workspace
// pointer is fetched on every call, so graph capture keeps its own buffer.
// The cache holds 1024 entries. This does not time kernels.

#include <ATen/native/utils/ParamsHash.h>

#include <list>
#include <mutex>
#include <unordered_map>

namespace {

struct HipblasltGemmConfig {
  int64_t m, n, k;
  int64_t lda, ldb, ldc, ldd;
  int64_t batch_count, stride_a, stride_b, stride_c;
  uint64_t workspace_bytes;
  int32_t a_type, b_type, c_type, d_type;
  int32_t compute_type, scale_type;
  int32_t transa, transb, epilogue;
  int32_t device, deterministic, sm_carveout;
};
static_assert(std::has_unique_object_representations_v<HipblasltGemmConfig>);

class HipblasltHeuristicCache {
 public:
  static HipblasltHeuristicCache& get() {
    static HipblasltHeuristicCache inst;
    return inst;
  }

  bool find(const HipblasltGemmConfig& key, cublasLtMatmulHeuristicResult_t* out) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = map_.find(key);
    if (it == map_.end()) {
      return false;
    }
    order_.splice(order_.begin(), order_, it->second);
    *out = it->second->second;
    return true;
  }

  void add(const HipblasltGemmConfig& key, const cublasLtMatmulHeuristicResult_t& result) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = map_.find(key);
    if (it != map_.end()) {
      it->second->second = result;
      order_.splice(order_.begin(), order_, it->second);
      return;
    }
    while (order_.size() >= kCapacity) {
      map_.erase(order_.back().first);
      order_.pop_back();
    }
    order_.emplace_front(key, result);
    map_.emplace(key, order_.begin());
  }

  void drop(const HipblasltGemmConfig& key) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = map_.find(key);
    if (it == map_.end()) {
      return;
    }
    order_.erase(it->second);
    map_.erase(it);
  }

 private:
  static constexpr size_t kCapacity = 1024;

  using Entry = std::pair<HipblasltGemmConfig, cublasLtMatmulHeuristicResult_t>;
  std::mutex mu_;
  std::list<Entry> order_;
  std::unordered_map<
      HipblasltGemmConfig,
      std::list<Entry>::iterator,
      at::native::ParamsHash<HipblasltGemmConfig>,
      at::native::ParamsEqual<HipblasltGemmConfig>>
      map_;
};

HipblasltGemmConfig hipblasltGemmConfig(
    cudaDataType_t abType,
    cudaDataType_t cType,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc, int64_t ldd,
    int64_t num_batches, int64_t stridea, int64_t strideb, int64_t stridec,
    cublasLtEpilogue_t epilogue,
    size_t workspace_size) {
  HipblasltGemmConfig key{};
  key.m = m;
  key.n = n;
  key.k = k;
  key.lda = lda;
  key.ldb = ldb;
  key.ldc = ldc;
  key.ldd = ldd;
  if (num_batches > 1) {
    key.batch_count = num_batches;
    key.stride_a = stridea;
    key.stride_b = strideb;
    key.stride_c = stridec;
  }
  key.workspace_bytes = workspace_size;
  key.a_type = abType;
  key.b_type = abType;
  key.c_type = cType;
  key.d_type = cType;
  key.compute_type = computeType;
  key.scale_type = scaleType;
  key.transa = transa;
  key.transb = transb;
  key.epilogue = epilogue;
  key.device = c10::cuda::current_device();
  key.deterministic = at::globalContext().deterministicAlgorithms() ? 1 : 0;
  key.sm_carveout = at::globalContext()._SMCarveout_EXPERIMENTAL().value_or(-1);
  return key;
}

} // namespace

#include "torch/csrc/jit/fuser/kernel_cache.h"

#include <unordered_map>
#include <mutex>
#include <cstdint>

namespace torch { namespace jit { namespace fuser {

static int64_t fusion_counter{0};
static std::unordered_map<int64_t, KernelSpec> specMap_;
static std::mutex mutex_;

int64_t store(std::shared_ptr<Graph> graph) {
  std::lock_guard<std::mutex> guard{mutex_};
  const auto key = fusion_counter++;
  specMap_.emplace(
  std::piecewise_construct
  , std::forward_as_tuple(key)
  , std::forward_as_tuple(key, graph));
  return key;
}

at::optional<KernelSpec&> retrieve(const int64_t key) { 
  std::lock_guard<std::mutex> guard{mutex_};
  auto it = specMap_.find(key);
  if (it == specMap_.end()) return c10::nullopt;
  return it->second;
}

} // namespace fuser
} // namespace jit
} // namespace torch
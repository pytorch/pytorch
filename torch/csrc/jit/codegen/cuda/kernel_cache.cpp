#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

// TODO: This class is dead at the moment, but we need to figure out a generic
// cacheing system that will suite our needs.

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

FusionExecutorCache::FusionExecutorCache(
    Fusion* fusion,
    CompileOptions options) {
  TORCH_INTERNAL_ASSERT(
      entry == nullptr,
      "At this time FusionExecutorCache only supports one entry.");
  entry = new FusionExecutor();
  entry->compileFusion(fusion, options);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

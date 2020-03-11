#include <torch/csrc/jit/codegen/cuda/fusion.h>

namespace torch {
namespace jit {
namespace fuser {

thread_local Fusion* FusionGuard::cur_fusion = nullptr;

} // namespace fuser
} // namespace jit
} // namespace torch

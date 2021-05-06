#include <torch/csrc/jit/mobile/nnc/registry.h>

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(NNCKernelRegistry, NNCKernel);

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch

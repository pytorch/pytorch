#include <torch/csrc/jit/mobile/nnc/registry.h>

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

C10_DEFINE_REGISTRY(NNCKernelRegistry, NNCKernel);

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch

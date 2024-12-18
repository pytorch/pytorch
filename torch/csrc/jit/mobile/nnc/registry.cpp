#include <torch/csrc/jit/mobile/nnc/registry.h>

namespace torch::jit::mobile::nnc {

C10_DEFINE_REGISTRY(NNCKernelRegistry, NNCKernel)

} // namespace torch::jit::mobile::nnc

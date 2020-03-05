#include <torch/csrc/jit/fuser/cuda/kernel.h>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void compileKernel(Fusion& fusion, CudaKernel& entry) {
  std::cout << "compiling kernel" << std::endl;
}

}}}} // namespace torch::jit::fuser::cuda

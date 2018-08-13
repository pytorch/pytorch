#include "torch/csrc/jit/fusers/fuser_interface.h"
#include "torch/csrc/jit/fusers/cpu/cpu_fuser_interface.h"

#ifdef USE_CUDA
  #include "torch/csrc/jit/fusers/cuda/cuda_fuser_interface.h"
#endif 

namespace torch { namespace jit { 

std::shared_ptr<CompiledFusionFunction> getFusionFunction(Node* fusion_group) {
  const auto device = fusion_group->i(attr::device);
  if (device == kCPUDevice) return getCPUFusionFunction(fusion_group);

  #ifdef USE_CUDA 
    return getCUDAFusionFunction(fusion_group);
  #endif // USE_CUDA

  throw std::runtime_error("CUDA device requested for fusion but USE_CUDA is undefined");  
}


} // namespace jit
} // namespace torch

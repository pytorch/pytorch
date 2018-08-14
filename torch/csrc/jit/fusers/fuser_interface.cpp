#include "torch/csrc/jit/fusers/fuser_interface.h"
#include "torch/csrc/jit/fusers/cpu/cpu_fuser_interface.h"

#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
  #include "torch/csrc/jit/fusers/cuda/cuda_fuser_interface.h"
#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM) 

namespace torch { namespace jit { 

std::shared_ptr<CompiledFusionFunction> getFusionFunction(Node* fusion_group) {
  const auto device = fusion_group->i(attr::device);
  if (device == kCPUDevice) return getCPUFusionFunction(fusion_group);

  #if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM) 
    return getCUDAFusionFunction(fusion_group);
  #endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)

  throw std::runtime_error("CUDA device requested for fusion but USE_CUDA is undefined");  
}


} // namespace jit
} // namespace torch

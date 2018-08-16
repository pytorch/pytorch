#include "torch/csrc/jit/fusers/fuser_interface.h"

#include "torch/csrc/jit/fusers/cpu/cpu_fuser_interface.h"

#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
  #include "torch/csrc/jit/fusers/cuda/cuda_fuser_interface.h"
#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM) 

namespace torch { namespace jit { 

std::shared_ptr<CompiledFusionFunction> getFusionFunction(Node* fusion_group) {
  const auto device = fusion_group->i(attr::device);
  if (device == kCPUDevice) {
    #if !(defined _WIN32)
      return getCPUFusionFunction(fusion_group);
    #endif // !(defined _WIN32)
    
    throw std::runtime_error("CPU fusion is not supported on this build"); 
  }

  #if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM) 
    return getCUDAFusionFunction(fusion_group);
  #endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)

  throw std::runtime_error("CUDA fusion is not supported on this build");  
}


} // namespace jit
} // namespace torch

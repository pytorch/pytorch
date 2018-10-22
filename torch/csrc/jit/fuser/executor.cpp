#include "torch/csrc/jit/fuser/executor.h"

#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "c10/util/Optional.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/kernel_cache.h"
#include "torch/csrc/jit/fuser/kernel_spec.h"
#include "torch/csrc/jit/fuser/compiler.h"

#if USE_CUDA_FUSER
  #include "torch/csrc/jit/fuser/cuda/interface.h"
#endif // USE_CUDA_FUSER

#if USE_CPU_FUSER
  #include "torch/csrc/jit/fuser/cpu/interface.h"
#endif // USE_CUDA_FUSER

#include <vector>
#include <tuple>
#include <stdexcept>
#include <algorithm>
#include <map>

namespace torch { namespace jit { namespace fuser {

void runFusion(
  const int64_t key
, Stack& stack) {
  // Short-circuits if fusion isn't enabled
  if (!canFuseOnCPU() && !canFuseOnGPU())
    throw std::runtime_error("Fusion not enabled.");

  // Acquires the FusionSpec
  auto maybe_spec = retrieve(key);
  if (!maybe_spec) 
    throw std::runtime_error("Failed to find fusion specification to run.");
  auto& spec = *maybe_spec;
  
  // Short-circuits if the spec isn't fusable
  if (!spec.isFusable()) 
    throw std::runtime_error("Non-fusable specification.");

  // Determines device to dispatch to
  // Acquires inputs from stack
  const auto inputs = fmap(last(stack, spec.nInputs()), [](const IValue& i) {
    return i.toTensor();
  });
  int32_t device = kCPUDevice;
  for (const auto& t : inputs) {
    const auto cur_device = t.device().index();
    if (cur_device < 0) continue;
    if (device < 0) device = cur_device;
    else if (device != cur_device) 
      throw std::runtime_error("Cannot fuse CUDA tensors on different devices.");
  }

  if (device >= 0 && canFuseOnGPU()) {
    #if USE_CUDA_FUSER
      const auto handle = cuda::getFusionHandle(spec, device);
      handle->run(stack);
    #endif // USE_CUDA_FUSER
  } else if (device == kCPUDevice && canFuseOnCPU()) {
    const auto handle = cpu::getFusionHandle(spec);
    handle->run(stack);
  } else {
    throw std::runtime_error("Fusion not enabled on requested device.");
  }

  
}

} // namespace fuser
} // namespace jit
} // namespace torch

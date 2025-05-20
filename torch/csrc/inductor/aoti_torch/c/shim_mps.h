#ifndef AOTI_TORCH_SHIM_MPS
#define AOTI_TORCH_SHIM_MPS

#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

typedef struct {
  std::shared_ptr<at::native::mps::MetalKernelFunction> kernelFunction;
} AOTIMetalKernelFunctionOpaque;

#ifdef __cplusplus
extern "C" {
#endif

using AOTIMetalKernelFunctionHandle = AOTIMetalKernelFunctionOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_set_arg(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AtenTensorHandle tensor);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOTI_TORCH_SHIM_MPS

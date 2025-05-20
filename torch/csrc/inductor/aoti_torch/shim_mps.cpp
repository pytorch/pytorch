#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

using namespace torch::aot_inductor;

AOTITorchError aoti_torch_mps_set_arg(
    AOTIMetalKernelFunctionHandle handle,
    unsigned idx,
    AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    handle->kernelFunction->setArg(idx, *t);
  });
}

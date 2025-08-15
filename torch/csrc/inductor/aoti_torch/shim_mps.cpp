#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

using namespace torch::aot_inductor;

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle handle,
    unsigned idx,
    AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto t = tensor_handle_to_tensor_pointer(tensor);
    if (t == nullptr) {
      throw std::runtime_error("Tensor is null.");
    }
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    func->setArg(idx, *t);
  });
}

AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle handle,
    unsigned idx,
    int64_t val) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    func->setArg(idx, val);
  });
}

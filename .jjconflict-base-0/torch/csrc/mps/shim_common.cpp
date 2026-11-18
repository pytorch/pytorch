#include <ATen/native/mps/MetalShaderLibrary.h>
#include <c10/util/Exception.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/c/shim.h>

AOTITorchError torch_mps_set_arg_bytes(
    AOTIMetalKernelFunctionHandle handle,
    unsigned idx,
    const void* ptr,
    uint64_t size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(ptr != nullptr, "Pointer is null.");
    TORCH_CHECK(
        size > 0 && size <= 4096,
        "size must be in (0, 4096], got ",
        size,
        ". Metal setBytes only supports transient data up to 4 KB. Pass larger data as a tensor.");
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    func->setArg(idx, ptr, size);
  });
}

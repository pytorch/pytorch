#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSDevice.h>


using namespace torch::aot_inductor;

AOTITorchError aoti_torch_mps_malloc(
    void** buffer,
    size_t num_bytes) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    id<MTLBuffer> metal_buffer = [device newBufferWithLength:num_bytes options:MTLResourceCPUCacheModeWriteCombined];
    if (!metal_buffer) {
      throw std::runtime_error("Failed to allocate memory on MPS device");
    }
    *buffer = (void*)metal_buffer;
  });
}

AOTITorchError aoti_torch_mps_free(
    void* ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    id<MTLBuffer> metal_buffer = (id<MTLBuffer>)ptr;
    [metal_buffer release];
  });
}

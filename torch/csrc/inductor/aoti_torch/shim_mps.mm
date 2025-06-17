#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSDevice.h>


using namespace torch::aot_inductor;

AOTITorchError aoti_torch_mps_malloc(
    void** buffer,
    size_t num_bytes) {
  if (num_bytes == 0) {
    *buffer = nullptr;
    return AOTI_TORCH_SUCCESS;
  }
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
      id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
      TORCH_CHECK(device, "Failed to get MPS device");
      id<MTLBuffer> metal_buffer = [device newBufferWithLength:num_bytes options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModeShared];
      TORCH_CHECK(metal_buffer, "Failed to allocate memory on MPS device");
      *buffer = (void*)metal_buffer;
  });
}

AOTITorchError aoti_torch_mps_free(
    void* ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto metal_buffer = (id<MTLBuffer>)ptr;
    [metal_buffer release];
  });
}


AOTITorchError
aoti_torch_mps_memcpy(void* buffer, size_t constant_offset, size_t bytes_read, size_t data_size, uint8_t* constants_start) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto metal_buffer = (id<MTLBuffer>)buffer;
    auto buffer_pointer = static_cast<uint8_t*>([metal_buffer contents]);
    memcpy(buffer_pointer + constant_offset, constants_start + bytes_read, data_size);
  });
}

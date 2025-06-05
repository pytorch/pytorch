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
    if (num_bytes != 0) {
      id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
      if (!device) {
        throw std::runtime_error("Failed to get MPS device");
      }
      id<MTLBuffer> metal_buffer = [device newBufferWithLength:num_bytes options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModeShared];
      if (!metal_buffer) {
        throw std::runtime_error("Failed to allocate memory on MPS device");
      }
      auto buffer_pointer = static_cast<uint8_t*>([metal_buffer contents]);
      std::cout << "aoti_torch_mps_malloc:: " << static_cast<void*>(buffer_pointer) << std::endl;
      *buffer = (void*)metal_buffer;
    }
  });
}

AOTITorchError aoti_torch_mps_free(
    void* ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    id<MTLBuffer> metal_buffer = (id<MTLBuffer>)ptr;
    [metal_buffer release];
  });
}


AOTITorchError
aoti_torch_mps_memcpy(void* buffer, size_t constant_offset, size_t bytes_read, size_t data_size, uint8_t* constants_start) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    id<MTLBuffer> metal_buffer = (id<MTLBuffer>)buffer;
    auto buffer_pointer = static_cast<uint8_t*>([metal_buffer contents]);
    std::cout << "aoti_torch_mps_memcpy:: " << static_cast<void*>(buffer) << " " << static_cast<void*>(buffer_pointer) << " " << constant_offset << std::endl;
    std::cout << std::endl;
    memcpy(buffer_pointer + constant_offset, constants_start + bytes_read, data_size);
    std::cout << "aoti_torch_mps_memcpy:: Buffer contents: ";
    const int32_t* int_ptr = reinterpret_cast<const int32_t*>(buffer_pointer + constant_offset);
    for (size_t i = 0; i < 16; i++) {  // 64 bytes = 16 integers
        std::cout << int_ptr[i] << " ";
    }
    std::cout << std::endl;
  });
}


AOTITorchError
aoti_torch_mps_print(AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto t = *tensor_handle_to_tensor_pointer(tensor);
    // id<MTLBuffer> metal_buffer = __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
    // auto buffer_pointer = static_cast<uint8_t*>([metal_buffer contents]);
    // std::cout << "mtl_setBuffer:: " << static_cast<void*>(buffer_pointer) << " " << t.storage_offset() << " Buffer contents: ";
    std::cout << "mtl_setBuffer:: " << t.storage().data() << " " << t.storage_offset() << " Buffer contents: ";
    auto data_ptr = static_cast<const uint8_t*>(t.storage().data());

    const int32_t* int_ptr = reinterpret_cast<const int32_t*>(data_ptr + t.storage_offset());
    for (size_t i = 0; i < 16; i++) {  // 64 bytes = 16 integers
        std::cout << int_ptr[i] << " ";
    }
    std::cout << std::endl;
  });
}

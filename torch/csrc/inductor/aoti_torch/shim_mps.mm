#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/mps/MPSStream.h>

using namespace torch::aot_inductor;

// The MPS allocator hands out real unified-memory addresses, so the returned
// pointer supports ordinary pointer arithmetic and CPU access. Metal APIs that
// need an MTLBuffer translate the pointer back via getMTLBuffer().
AOTITorchError aoti_torch_mps_malloc(
    void** buffer,
    size_t num_bytes) {
  if (num_bytes == 0) {
    *buffer = nullptr;
    return AOTI_TORCH_SUCCESS;
  }
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto data_ptr = at::mps::getIMPSAllocator()->allocate(num_bytes);
      TORCH_CHECK(data_ptr, "Failed to allocate ", num_bytes, " bytes on MPS device");
      *buffer = data_ptr.get();
      data_ptr.release_context();
  });
}

AOTITorchError aoti_torch_mps_free(
    void* ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (ptr) {
      at::mps::getIMPSAllocator()->raw_deleter()(ptr);
    }
  });
}

AOTITorchError
aoti_torch_mps_memcpy(void* buffer, size_t constant_offset, size_t bytes_read, size_t data_size, uint8_t* constants_start) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    memcpy(static_cast<uint8_t*>(buffer) + constant_offset, constants_start + bytes_read, data_size);
  });
}

AOTITorchError
aoti_torch_mps_copy_buffer(void* src_buffer, void* dst_buffer, size_t data_size, size_t src_offset, size_t dst_offset) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* allocator = at::mps::getIMPSAllocator();
    auto src_mtl_buffer = __builtin_bit_cast(id<MTLBuffer>, allocator->getMTLBuffer(src_buffer));
    auto dst_mtl_buffer = __builtin_bit_cast(id<MTLBuffer>, allocator->getMTLBuffer(dst_buffer));
    TORCH_CHECK(src_mtl_buffer && dst_mtl_buffer,
                "aoti_torch_mps_copy_buffer: pointers must be MPSAllocator base data pointers");

    auto* stream = at::mps::getCurrentMPSStream();
    auto profile_id = at::mps::getMPSProfiler().beginProfileCopy(
        src_buffer, dst_buffer, at::OptionalTensorRef(), at::OptionalTensorRef(), data_size, true);
    stream->copy_and_sync(src_mtl_buffer, dst_mtl_buffer, data_size, src_offset, dst_offset, true, profile_id);
  });
}

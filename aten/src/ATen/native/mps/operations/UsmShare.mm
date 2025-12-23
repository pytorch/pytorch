#include <ATen/ATen.h>
#include <ATen/mps/MPSDevice.h>
#include <Metal/Metal.h>
#include <unistd.h>

namespace at::native {

// Implementation for MPS backend
// src passed by value to match dispatcher signature
Tensor usm_share_from_mps(const Tensor& self, c10::Storage src) {
  void* ptr = src.data_ptr().get();
  size_t size = src.nbytes();

  TORCH_CHECK(
      src.device().is_cpu(),
      "usm_share_from_mps: source storage must be on CPU, got: ",
      src.device());
  TORCH_CHECK(
      self.device().type() == c10::DeviceType::MPS,
      "usm_share_from_mps: target device must be MPS, got: ",
      self.device());

  // Handle empty case
  if (size == 0) {
    return at::empty_like(self, self.options());
  }

  // 1. Check Alignment (Critical for Metal NoCopy)
  int pageSize = getpagesize();
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();

  // 2. Create NoCopy Buffer
  // Length must be page-aligned for newBufferWithBytesNoCopy
  size_t aligned_size = (size + pageSize - 1) & ~(pageSize - 1);
  id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:ptr
                                                   length:aligned_size
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
  
  TORCH_CHECK(buffer, "usm_share_from_mps: Failed to create MTLBuffer. "
                      "Ensure size is page-aligned and pointer is page-aligned.");
  
  void* buffer_ptr = (void*)buffer;
  
  // Increment ref count of src to prevent it from being freed while the new
  // storage shares its data. The ref count will be decremented in the deleter.
  c10::StorageImpl* src_impl = src.unsafeGetStorageImpl();
  c10::raw::intrusive_ptr::incref(src_impl);

  struct DeleterContext {
    void* buffer;
    c10::StorageImpl* src_impl;
  };

  auto* deleter_context = new DeleterContext{buffer_ptr, src_impl};

  c10::DeleterFnPtr deleter = [](void* ctx) {
    auto* context = static_cast<DeleterContext*>(ctx);
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(context->buffer);
    [buffer release];
    c10::raw::intrusive_ptr::decref(context->src_impl);
    delete context;
  };

  // 3. Create StorageImpl
  c10::Allocator* mps_allocator = at::mps::GetMPSAllocator();
  TORCH_CHECK(mps_allocator, "usm_share_from_mps: MPS Allocator is null");

  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size,
      c10::DataPtr(buffer_ptr, deleter_context, deleter, self.device()),
      mps_allocator,
      false);

  return at::empty({0}, self.options()).set_(std::move(storage_impl));
}

} // namespace at::native

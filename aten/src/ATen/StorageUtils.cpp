#include <ATen/Functions.h>
#include <ATen/MapAllocator.h>
#include <ATen/StorageUtils.h>
#include <c10/core/TensorOptions.h>

namespace at {

C10_EXPORT c10::intrusive_ptr<c10::StorageImpl> new_shm_fd_storage(
    size_t size) {
  int flags = ALLOCATOR_MAPPED_SHAREDMEM | ALLOCATOR_MAPPED_EXCLUSIVE |
      ALLOCATOR_MAPPED_KEEPFD | ALLOCATOR_MAPPED_UNLINK;
  std::string handle = NewProcessWideShmHandle();
  auto sptr = MapAllocator::makeDataPtr(
      handle, flags, size * sizeof(uint8_t), nullptr);
  return c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size,
      std::move(sptr),
      /*allocator=*/nullptr,
      /*resizable=*/false);
}

C10_EXPORT void storage_copy(
    c10::Storage& dst,
    const c10::Storage& src,
    bool non_blocking) {
  auto dst_options = c10::TensorOptions().device(dst.device()).dtype(at::kByte);
  auto dst_t = at::empty({0}, dst_options).set_(dst);

  auto src_options = c10::TensorOptions().device(src.device()).dtype(at::kByte);
  auto src_t = at::empty({0}, src_options).set_(src);
  dst_t.copy_(src_t, non_blocking);
}

C10_EXPORT void share_memory_(TensorBase& t) {
  if (t.device() != at::kCPU) {
    return;
  }

  const at::Storage& origStorage = t.storage();

  if (MapAllocator::fromDataPtr(origStorage.data_ptr()) != nullptr) {
    // already shared
    return;
  }
  at::Storage newStorage(new_shm_fd_storage(origStorage.nbytes()));
  storage_copy(newStorage, origStorage);

  // Replace the old data_ptr and allocator with the new ones
  c10::StorageImpl* origStorageImpl = origStorage.unsafeGetStorageImpl();
  c10::StorageImpl* newStorageImpl = newStorage.unsafeGetStorageImpl();
  origStorageImpl->set_data_ptr(std::move(newStorageImpl->mutable_data_ptr()));
  origStorageImpl->set_allocator(newStorageImpl->allocator());
}

C10_EXPORT c10::Storage usm_share(
    const c10::Storage& src,
    const c10::Device& device) {
  // 1. Validate Source
  TORCH_CHECK(src.device().is_cpu(), "usm_share: source storage must be on CPU");
  
  // 2. Create Dummy Tensor on Target Device to trigger Dispatch
  // We use size {0} so it shouldn't allocate significant memory,
  // but carries the backend info (HIP/MPS/CUDA/etc.)
  auto dst_options = at::TensorOptions().dtype(at::kByte).device(device);
  auto dst_dummy = at::empty({0}, dst_options);

  // 3. Call Native Function via Dispatcher (Storage-based)
  // The dispatcher looks at 'dst_dummy' to decide which backend file to use.
  Tensor result_tensor = at::_usm_share_from(dst_dummy, src);

  // 4. Unwrap Storage
  return result_tensor.storage();
}

} // namespace at

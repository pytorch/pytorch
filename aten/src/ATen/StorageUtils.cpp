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
      handle.c_str(), flags, size * sizeof(uint8_t), nullptr);
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

} // namespace at

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/EmptyTensor.h>
#include <ATen/TensorUtils.h>

#include <c10/core/CPUAllocator.h>
#include <c10/core/SymBool.h>

#include <utility>


namespace at::native {

// TODO: make all operations that resize given outputs use this function
//   for consistency and maintainability.
//   Some operations like `cat` might not be able to make the use of
//   resize_output directly. For more details to understand how it works in `cat`,
//   see https://github.com/pytorch/pytorch/pull/62560#discussion_r687363362
// Resizes outputs
// Functions accepting output tensors, like with the "out" kwarg, should
//   call this function to handle resizing their output tensor.
// Issues a warning if the output tensor has one or more elements and
//   needs resizing
// NOTE: In the future the warning will become an error
// Returns a bool saying whether or not the resize actually happened or not
TORCH_API bool resize_output(const Tensor& output, IntArrayRef shape);
// WARNING: Do NOT call this directly. If you are resizing an output and want
// to support dynamic shapes call at::resize__symint and resize_output_check_symint.
// For more details, see: https://github.com/pytorch/pytorch/pull/111530/files#r1365845272
TORCH_API bool resize_output_symint(const Tensor& output, SymIntArrayRef shape);

// Utility for resize_output
//  Returns a bool saying resize should happen or not and
//  raises a warning if resizing for one or more elements
TORCH_API bool resize_output_check(const Tensor& output, IntArrayRef shape);
TORCH_API bool resize_output_check_symint(const Tensor& output, SymIntArrayRef shape);

TORCH_API void resize_bytes_cpu(StorageImpl* storage, size_t size_bytes);
TORCH_API void resize_bytes_meta(StorageImpl* storage, c10::SymInt size_bytes);
TORCH_API void resize_bytes_nocuda(const Storage& storage, const c10::SymInt& size_bytes);

inline void maybe_resize_storage_cpu(TensorImpl* self, size_t new_size_bytes) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in cuda/Resize.h)
  if (self->numel() == 0) {
    return;
  }

  const Storage& storage = self->unsafe_storage();
  if (!storage) {
    auto new_storage = c10::make_intrusive<StorageImpl>(
        StorageImpl::use_byte_size_t(),
        new_size_bytes,
        c10::GetCPUAllocator(),
        true);
    self->set_storage_keep_dtype(std::move(new_storage));
  } else if (new_size_bytes > storage.nbytes()) {
    resize_bytes_cpu(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

TORCH_API TensorImpl* resize_impl_cpu_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride,
    bool resize_storage = true);

template <typename T>
T maybe_convert_symint(c10::SymInt) = delete;

template <>
inline c10::SymInt maybe_convert_symint(c10::SymInt x) { return x; }

template <>
inline int64_t maybe_convert_symint(c10::SymInt x) { return x.guard_int(__FILE__, __LINE__); }

template <typename T>
inline void checkInBoundsForStorage(
    ArrayRef<T> size,
    ArrayRef<T> stride,
    T storage_offset,
    const caffe2::TypeMeta& data_type,
    const Storage& new_storage) {
  T storage_size_bytes, storage_size_plus_offset_bytes;
  if (stride.data()) {
    storage_size_bytes =
        at::detail::computeStorageNbytes(size, stride, data_type.itemsize());
    storage_size_plus_offset_bytes = at::detail::computeStorageNbytes(
        size, stride, data_type.itemsize(), storage_offset);
  } else {
    storage_size_bytes =
        at::detail::computeStorageNbytesContiguous(size, data_type.itemsize());
    storage_size_plus_offset_bytes = at::detail::computeStorageNbytesContiguous(
        size, data_type.itemsize(), storage_offset);
  }
  // It's ok to always evaluate to False for this early return for SymInts because
  // (1) maybe_convert_symint below only installs guard for int64_t case
  // (2) we check for this condition in the TORCH_MAYBE_SYM_CHECK below
  if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_eq(storage_size_bytes, 0))) {
    // NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
    return;
  }
  T new_storage_size_bytes = maybe_convert_symint<T>(new_storage.sym_nbytes());
  TORCH_MAYBE_SYM_CHECK(
      sym_eq(storage_size_bytes, 0) || sym_le(storage_size_plus_offset_bytes, new_storage_size_bytes),
      "setStorage: sizes ",
      size,
      ", strides ",
      stride,
      ","
      " storage offset ",
      storage_offset,
      ", and itemsize ",
      data_type.itemsize(),
      " requiring a storage size of ",
      storage_size_plus_offset_bytes,
      " are out of bounds for storage of size ",
      new_storage_size_bytes);
}

template <typename T>
inline void checkSetStorage(Tensor& result, Storage storage, T storage_offset,
                                   ArrayRef<T> size, ArrayRef<T> stride, bool check_offset_in_bounds = true) {
  // FIXME: stride should be optional
  if (stride.data()) {
    TORCH_CHECK(size.size() == stride.size(), "unequal size length (", size.size(),
                                              ") and stride length (", stride.size(), ")");
  }

#ifdef DEBUG
  TORCH_CHECK(size.size() <= INT_MAX, "size length (", size.size(), ") greater than INT_MAX");
#endif

  // storageOffset
  TORCH_CHECK(
      storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);

  // set_storage_{device} (except set_storage_meta__symint)
  // will (unsafely) set the storage offset and then call resize_impl that
  // handles resizing the storage However, resize_impl will only resize the
  // storage if the sizes/strides changed. For the case that the sizes/strides
  // remain unchanged, the storage offset is not properly validated, so we do
  // that here.
  if (check_offset_in_bounds) {
    auto result_tensor_impl = result.unsafeGetTensorImpl();
    bool size_unchanged = result_tensor_impl->generic_sizes<T>() == size;
    bool stride_unchanged = stride.data()
        ? result_tensor_impl->generic_strides<T>() == stride
        : true;
    if (size_unchanged && stride_unchanged) {
      checkInBoundsForStorage(
          size, stride, storage_offset, result.dtype(), storage);
    }
  }

  // storage: note this can't be replaced with result.set_(storage) as the semantics of that
  // function is to set the tensor size to be equal to the size of the storage.
  if (!result.storage().is_alias_of(storage)) {
    // Caffe2 might have tensors whose storages are null, but we
    // don't allow it in PyTorch.
    TORCH_INTERNAL_ASSERT(storage);
    TORCH_INTERNAL_ASSERT(result.storage());

    // We used to allow this, but this breaks device caching.
    // Let's put an actual error message for this one.
    TORCH_CHECK(result.storage().device() == storage.device(),
                "Attempted to set the storage of a tensor on device \"", result.storage().device(),
                "\" to a storage on different device \"", storage.device(),
                "\".  This is no longer allowed; the devices must match.");
    result.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(storage));
  }
}

/**
 * Set self's sizes, strides, and storage_offset.
 * (size, stride, storage_offset) must be in bounds for self's storage.
 */
template <typename T>
inline void setStrided(
    const Tensor& self,
    ArrayRef<T> size,
    ArrayRef<T> stride,
    T storage_offset) {
  TORCH_CHECK(size.size() == stride.size(), "mismatch in length of strides and shape");
  for (const auto& val : stride) {
    TORCH_CHECK(val >= 0,
                "as_strided: Negative strides are not supported at the moment, "
                "got strides: ", stride);
  }

  auto* self_ = self.unsafeGetTensorImpl();
  checkInBoundsForStorage(
      size, stride, storage_offset, self_->dtype(), self_->storage());

  /* storage offset */
  TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
  self_->set_sizes_and_strides(size, stride, storage_offset);
}

} // namespace at::native

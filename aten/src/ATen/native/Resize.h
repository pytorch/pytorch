#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/TensorUtils.h>

#include <c10/core/CPUAllocator.h>

namespace at { namespace native {

using ResizeImplFn = TensorImpl*(*)(TensorImpl*, IntArrayRef, c10::optional<IntArrayRef>, bool);
using SelectSizeFn = uint64_t(*)(TensorImpl*, IntArrayRef, c10::optional<IntArrayRef>);
using MaybeResizeFn = void(*)(TensorImpl*, uint64_t);

// Template implementation for `resize_`.
// Implementors should call this function with the core
// resizing implementation as template parameter.
template <ResizeImplFn Impl>
void resize_template(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<IntArrayRef> strides,
    c10::optional<MemoryFormat> optional_memory_format,
    bool resize_storage) {
  TORCH_INTERNAL_ASSERT(!(strides.has_value() && optional_memory_format.has_value()));
  if (self.has_names()) {
    resize_named_tensor_(self, size, optional_memory_format);
    return;
  }
  auto* self_ = self.unsafeGetTensorImpl();
  Impl(self_, size, strides, resize_storage);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
}

// Core logic for resize implementations:
// 1. Short-circuit if we are not really changing anything
// 2. Select the new storage size
// 3. Actually resize (allocation may happen)
template <MaybeResizeFn MaybeResize, SelectSizeFn SelectFn>
TensorImpl* resize_impl_template_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool resize_storage = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  uint64_t storage_size = SelectFn(self, size, stride);

  if (resize_storage) {
    MaybeResize(self, storage_size);
  }

  return self;
}

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
TORCH_API bool resize_output(
    const Tensor& output,
    IntArrayRef shape,
    c10::optional<IntArrayRef> strides = c10::nullopt);

// Utility for resize_output
//  Returns a bool saying resize should happen or not and
//  raises a warning if resizing for one or more elements
TORCH_API bool resize_output_check(const Tensor& output, IntArrayRef shape);

TORCH_API void resize_bytes_cpu(StorageImpl* storage, size_t size_bytes);

static inline void maybe_resize_storage_cpu(TensorImpl* self, uint64_t new_size) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in cuda/Resize.h)
  if (new_size == 0) {
    return;
  }

  const auto new_size_bytes_i =
      (new_size + self->storage_offset()) * self->dtype().itemsize();
  TORCH_CHECK(!overflows<size_t>(new_size_bytes_i), "Requested storage size (",
              new_size_bytes_i, ") cannot be represented as a size_t");
  const auto new_size_bytes = static_cast<size_t>(new_size_bytes_i);

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

inline uint64_t select_storage_size_default(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride) {
  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    storage_size = storage_size_for(size, *stride);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  return storage_size;
}

TensorImpl* resize_impl_cpu_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool resize_storage = true);

// Try to reuse the storage by ignoring the given strides, if
// necessary. There are 3 cases:
// 1. Using the requested size and strides fit inside the existing
//    storage -> no allocation needed!
// 2. It only fits in the existing storage if we don't use the
//    requested strides -> no allocation needed!
// 3. We have to allocate memory -> allocation needed!
inline uint64_t select_storage_size_tryreuse(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride) {
  const size_t storage_nbytes = self->storage().nbytes();

  // Defines how we are allocating/reusing memory.
  // CONTIGUOUS has lower priority than STRIDED.
  enum Strategy { CONTIGUOUS = 0, STRIDED, UNDEF };
  // Checks that will be used to decide what strategy we will use
  struct Checks {
    bool may_reuse;
    bool did_overflow;
  };

  auto check_allocatable = [&](const uint64_t storage_size) -> Checks {
    auto byte_size = (storage_size + self->storage_offset()) * self->dtype().itemsize();
    auto bytes = static_cast<size_t>(byte_size);
    return Checks{bytes <= storage_nbytes, overflows<size_t>(byte_size)};
  };

  // Keep track of which strategy we are going to use for reusing
  // the storage (if that's possible).
  Strategy selected = UNDEF;
  // If we can't, we pick an allocation strategy.
  Strategy fallback = UNDEF;

  std::array<uint64_t, 2> storage_size{0, 0};

  // Computation of contiguous storage size is necessary, here.
  // Calling 'self->set_size_contiguous' already does it, but we
  // still don't know at this point.
  storage_size[CONTIGUOUS] = 1;
  for (auto s : size) {
    storage_size[CONTIGUOUS] *= s;
  }

  auto c = check_allocatable(storage_size[CONTIGUOUS]);
  selected = (!c.did_overflow && c.may_reuse) ? CONTIGUOUS : selected;
  fallback = !c.did_overflow ? CONTIGUOUS : fallback;

  // Use STRIDED strategy only if we have strides.
  if (stride.has_value()) {
    storage_size[STRIDED] = storage_size_for(size, stride.value());
    auto c = check_allocatable(storage_size[STRIDED]);
    selected = (!c.did_overflow && c.may_reuse) ? STRIDED : selected;
    fallback = !c.did_overflow ? STRIDED : fallback;
  }

  if (selected == UNDEF) {
    // If both `selected` and `alloc_strategy` are `UNDEF`, it means
    // that we've overflowed on all strategies.
    TORCH_CHECK(
        fallback != UNDEF,
        "Requested storage size (",
        storage_size[CONTIGUOUS],
        ") cannot be represented as a size_t");
    selected = fallback;
  }

  switch (selected) {
    case Strategy::STRIDED:
      self->set_sizes_and_strides(size, stride.value());
      break;
    case Strategy::CONTIGUOUS:
      self->set_sizes_contiguous(size);
      break;
    default:
      break;
  }

  return storage_size[selected];
}

template <MaybeResizeFn MaybeFn>
TensorImpl* resize_impl_tryreuse_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool resize_storage = true) {
  return resize_impl_template_<MaybeFn, &select_storage_size_tryreuse>(
      self, size, stride);
}

static inline void checkInBoundsForStorage(
    IntArrayRef size,
    IntArrayRef stride,
    int64_t storage_offset,
    const caffe2::TypeMeta data_type,
    const Storage& new_storage) {
  int64_t storage_size_bytes =
      at::detail::computeStorageNbytes(size, stride, data_type.itemsize());
  int64_t storage_offset_bytes = storage_offset * data_type.itemsize();
  if (storage_size_bytes == 0) {
    // NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
    return;
  }
  int64_t new_storage_size_bytes = new_storage.nbytes();
  TORCH_CHECK(
      storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
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
      storage_size_bytes + storage_offset_bytes,
      " are out of bounds for storage of size ",
      new_storage_size_bytes);
}

static inline void checkSetStorage(Tensor& result, Storage storage, int64_t storage_offset,
                                   IntArrayRef size, IntArrayRef stride) {
  // FIXME: stride should be optional
  if (stride.data()) {
    TORCH_CHECK(size.size() == stride.size(), "unequal size length (", size.size(),
                                              ") and stride length (", stride.size(), ")");
  }

#ifdef DEBUG
  TORCH_CHECK(size.size() <= INT_MAX, "size length (", size.size(), ") greater than INT_MAX");
#endif

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
    result.unsafeGetTensorImpl()->set_storage_keep_dtype(storage);
  }

  // storageOffset
  TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
}

/**
 * Set self's sizes, strides, and storage_offset.
 * (size, stride, storage_offset) must be in bounds for self's storage.
 */
inline void setStrided(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    int64_t storage_offset) {
  TORCH_CHECK(size.size() == stride.size(), "mismatch in length of strides and shape");
  auto* self_ = self.unsafeGetTensorImpl();
  checkInBoundsForStorage(
      size, stride, storage_offset, self_->dtype(), self_->storage());

  /* storage offset */
  TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
  self_->set_storage_offset(storage_offset);

  /* size and stride */
  if (self_->sizes() == size && self_->strides() == stride) {
    return;
  }
  for (auto val : stride) {
    TORCH_CHECK(val >= 0,
                "as_strided: Negative strides are not supported at the moment, "
                "got strides: ", stride);
  }
  self_->set_sizes_and_strides(size, stride);
}

}}

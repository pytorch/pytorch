#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/resize.h>
#include <ATen/ops/_resize_output.h>
#include <ATen/ops/_resize_output_native.h>
#endif

namespace at::native {

// Returns true if resize is necessary
template <typename T>
bool _resize_output_check(const Tensor& output, ArrayRef<T> shape) {
  // Tests for resizing of tensors with one or more elements
  if (at::symint::sizes<T>(output).equals(shape)) {
    return false;
  }
  if (at::symint::numel<T>(output) != 0) {
    TORCH_WARN(
      "An output with one or more elements was resized since it had ",
      "shape ", at::symint::sizes<T>(output), ", which does not match the required ",
      "output shape ", shape, ". ",
      "This behavior is deprecated, and in a future PyTorch release outputs ",
      "will not be resized unless they have zero elements. You can explicitly ",
      "reuse an out tensor t by resizing it, inplace, to zero elements with ",
      "t.resize_(0).");
  }
  return true;
}

bool resize_output_check(const Tensor& output, IntArrayRef shape) {
  return _resize_output_check(output, shape);
}

bool resize_output_check_symint(const Tensor& output, SymIntArrayRef shape) {
  return _resize_output_check(output, shape);
}

static void native_resize_(const Tensor& output, IntArrayRef shape) {
  native::resize_(output, shape);
}

static void native_resize_(const Tensor& output, SymIntArrayRef shape) {
  native::resize__symint(output, shape);
}

template <typename T>
bool _resize_output(const Tensor& output, ArrayRef<T> shape) {
  if (_resize_output_check<T>(output, shape)) {
    // avoid a redispatch for cpu and cuda.
    // TODO: when resize_cuda_ is re-written to be unified with resize_,
    // we can provide the same benefit for cuda.
    //
    // TODO(#61485): functorch wrapped tensors should not go through the
    // fast path. This is a hack, longer term solutions are in the issue
    if (output.is_cpu() && !isTensorSubclassLike(output)) {
      native_resize_(output, shape);
    } else {
      at::symint::resize_<T>(output, shape);
    }
    return true;
  } else {
    return false;
  }
}

bool resize_output(const Tensor& output, IntArrayRef shape) {
  return _resize_output(output, shape);
}

bool resize_output_symint(const Tensor& output, SymIntArrayRef shape) {
  return _resize_output(output, shape);
}

const Tensor& _resize_output_(const Tensor& self, IntArrayRef shape, c10::Device device) {
  TORCH_CHECK(self.device() == device, "out Tensor doesn't have the correct device set");
  at::native::resize_output(self, shape);
  return self;
}

void resize_bytes_cpu(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");

  at::DataPtr new_data;
  if (size_bytes != 0) {
    new_data = storage->allocator()->allocate(size_bytes);
  }
  const at::DataPtr& old_data = storage->data_ptr();
  const auto old_capacity = storage->nbytes();
  const auto copy_capacity = std::min(size_bytes, old_capacity);
  if (old_data != nullptr && copy_capacity > 0) {
    memcpy(new_data.get(), old_data.get(), copy_capacity);
  }
  storage->set_data_ptr_noswap(std::move(new_data));
  storage->set_nbytes(size_bytes);
}

// Call the sparse implementation in SparseTensor.cpp directly.
// A dynamic dispatch here is NOT necessary, so I didn't put
// this function in native_functions.yaml
const Tensor& resize_as_sparse_(const Tensor& self, const Tensor& src);

// TODO(VitalyFedyunin): Move it to HTML docs.
//
// Strides of the output tensor of `resize_as_` operator is defined by input
// tensor strides and the value of memory_format argument.
//
// If memory_format is omitted and input tensor have the same shape as output
// tensor, strides of the output will remain unchanged. Strides going to be
// set to contiguous if shapes are different.
//
// If memory_format is equals to MemoryFormat::Contiguous (torch.contiguous_format)
// output tensor will have contiguous strides.
//
// If memory_format is equal to MemoryFormat::ChannelsLast (torch.channels_last)
// and input tensor is 4D, output tensor will have channels last memory layout.
//
// If memory_format is equal to MemoryFormat::Preserve (torch.preserve_format)
// output tensor will be defined by strides of the input tensor, following
// memory format preservation rule:
//
//  - If input tensor strides are in channels last format, output tensor will
//    have channels last memory layout.
//
//  - Otherwise, output tensor will have contiguous memory layout.
//
const Tensor& resize_as_(
    const Tensor& self,
    const Tensor& the_template,
    std::optional<MemoryFormat> optional_memory_format) {
  if (self.is_sparse() && the_template.is_sparse()) {
    TORCH_CHECK(
        !optional_memory_format.has_value(),
        "Unsupported memory format for sparse tensor resize_as_ :",
        optional_memory_format.value());
    return at::native::resize_as_sparse_(self, the_template);
  }
  const Tensor& result = self.resize_(the_template.sizes());
  if (optional_memory_format.has_value()) {
    auto memory_format = optional_memory_format.value();
    if (memory_format == MemoryFormat::Preserve) {
      memory_format = the_template.suggest_memory_format();
    }
    self.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  }
  namedinference::propagate_names(result, the_template);
  return result;
}


void resize_bytes_meta(StorageImpl* storage, c10::SymInt size_bytes) {
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
  storage->set_nbytes(std::move(size_bytes));
}

static void maybe_resize_storage_meta(TensorImpl* self, c10::SymInt new_size_bytes) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.h)
  if (self->sym_numel() == 0) {
    return;
  }

  const Storage& storage = self->unsafe_storage();
  if (!storage) {
    TORCH_INTERNAL_ASSERT(0, "NYI, this should only be Caffe2");
  } else if (new_size_bytes > storage.sym_nbytes()) {
    resize_bytes_meta(storage.unsafeGetStorageImpl(), std::move(new_size_bytes));
  }
}

static void _maybe_resize_storage(TensorImpl* self, int64_t new_size_bytes) {
  maybe_resize_storage_cpu(self, new_size_bytes);
}

static void _maybe_resize_storage(TensorImpl* self, c10::SymInt new_size_bytes) {
  if (self->is_cpu()) {
    maybe_resize_storage_cpu(self, new_size_bytes.expect_int());
    return;
  }
  TORCH_INTERNAL_ASSERT(self->is_meta());
  maybe_resize_storage_meta(self, std::move(new_size_bytes));
}

template <typename T>
TensorImpl* _resize_impl_(
    TensorImpl* self,
    ArrayRef<T> size,
    at::OptionalArrayRef<T> stride,
    bool resize_storage) {
  if (self->generic_sizes<T>() == size && (!stride || self->generic_strides<T>() == stride.value())) {
    return self;
  }

  const auto itemsize = self->dtype().itemsize();
  const auto storage_offset = self->generic_storage_offset<T>();
  T storage_size = T(1);
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(
        size, *stride, itemsize, storage_offset);
  } else {
    self->generic_set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(
        size, itemsize, storage_offset);
  }

  if (resize_storage) {
    _maybe_resize_storage(self, std::move(storage_size));
  }

  return self;
}

TensorImpl* resize_impl_cpu_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride,
    bool resize_storage) {
  return _resize_impl_(self, size, stride, resize_storage);
}

template <typename T>
const Tensor& _resize_(
    const Tensor& self,
    ArrayRef<T> size,
    std::optional<MemoryFormat> optional_memory_format) {
  auto* self_ = self.unsafeGetTensorImpl();
  int64_t old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().sym_nbytes().maybe_as_int().value_or(-1) : 0;
  // NOLINTNEXTLINE(bugprone-argument-comment)
  _resize_impl_<T>(self_, size, /*strides=*/c10::nullopt, true);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory() && old_storage_nbytes != -1)) {
    at::native::fill_resize_deterministic_(self, old_storage_nbytes);
  }
  return self;
}

const Tensor& resize_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  return _resize_(self, size, optional_memory_format);
}

const Tensor& resize__symint(
    const Tensor& self,
    c10::SymIntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  TORCH_INTERNAL_ASSERT(!self.has_names())
  return _resize_(self, size, optional_memory_format);
}

void resize_bytes_nocuda(const Storage& storage, c10::SymInt newsize) {
  // handles all devices except cuda (which needs to be in a different .so)
  c10::DeviceType device_type = storage.device_type();
  if (device_type == at::kCPU) {
    at::native::resize_bytes_cpu(storage.unsafeGetStorageImpl(), newsize.expect_int());
  } else if (device_type == at::kMeta) {
    at::native::resize_bytes_meta(storage.unsafeGetStorageImpl(), newsize);
  } else if (device_type == at::kPrivateUse1) {
    at::GetPrivateUse1HooksInterface()->resizePrivateUse1Bytes(
        storage, newsize.expect_int());
  } else if (device_type == at::kXPU || device_type == at::kHPU) {
    ptrdiff_t size_bytes_i = newsize.expect_int();
    TORCH_CHECK(
        !c10::overflows<int64_t>(size_bytes_i),
        "Requested storage size (",
        size_bytes_i,
        ") cannot be represented as a int64_t");
    const auto size_bytes = static_cast<int64_t>(size_bytes_i);
    void* original_data_ptr = storage.data_ptr().get();

    auto src_option =
        c10::TensorOptions().device(storage.device()).dtype(at::kByte);
    auto src_tensor = at::empty({0}, src_option).set_(storage);
    src_tensor.resize_({size_bytes});

    // When using resize_ to replace resize_bytes_xxx, in some cases
    // the original data_ptr is still returned, which is an inconsistent
    // behavior when compared to resize_bytes_xxx. For these cases,
    // an additional memory copy and update for storage are required.
    if (original_data_ptr == src_tensor.storage().data_ptr().get()) {
      auto new_tensor = at::empty(src_tensor.sizes(), src_tensor.options());
      new_tensor.copy_(src_tensor);
      storage.set_data_ptr_noswap(
          std::move(new_tensor.storage().mutable_data_ptr()));
      storage.unsafeGetStorageImpl()->set_allocator(
          new_tensor.storage().unsafeGetStorageImpl()->allocator());
      storage.set_nbytes(new_tensor.storage().nbytes());
    }
  } else {
    TORCH_CHECK(
        false,
        "UntypedStorage.resize_: got unexpected device type ",
        device_type);
  }
}

} // namespace at::native

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
#endif

namespace at { namespace native {

// Returns true if resize is necessary
bool resize_output_check(const Tensor& output, IntArrayRef shape) {
  // Tests for resizing of tensors with one or more elements
  if (output.sizes().equals(shape)) {
    return false;
  }
  if (output.numel() != 0) {
    TORCH_WARN(
      "An output with one or more elements was resized since it had ",
      "shape ", output.sizes(), ", which does not match the required ",
      "output shape ", shape, ". ",
      "This behavior is deprecated, and in a future PyTorch release outputs ",
      "will not be resized unless they have zero elements. You can explicitly ",
      "reuse an out tensor t by resizing it, inplace, to zero elements with ",
      "t.resize_(0).");
  }
  return true;
}

// Returns true if resize is necessary
bool resize_output_check_symint(const Tensor& output, SymIntArrayRef shape) {
  // Tests for resizing of tensors with one or more elements
  if (output.sym_sizes().equals(shape)) {
    return false;
  }
  if (output.numel() != 0) {
    TORCH_WARN(
      "An output with one or more elements was resized since it had ",
      "shape ", output.sizes(), ", which does not match the required ",
      "output shape ", shape, ". ",
      "This behavior is deprecated, and in a future PyTorch release outputs ",
      "will not be resized unless they have zero elements. You can explicitly ",
      "reuse an out tensor t by resizing it, inplace, to zero elements with ",
      "t.resize_(0).");
  }
  return true;
}

template <typename T>
bool resize_output(const Tensor& output, T shape) {
  if (resize_output_check(output, shape)) {
    // avoid a redispatch for cpu and cuda.
    // TODO: when resize_cuda_ is re-written to be unified with resize_,
    // we can provide the same benefit for cuda.
    //
    // TODO(#61485): functorch wrapped tensors should not go through the
    // fast path. This is a hack, longer term solutions are in the issue
    if (output.is_cpu() && !isTensorSubclassLike(output)) {
      at::native::resize_(output, shape);
    } else {
      output.resize_(shape);
    }
    return true;
  } else {
    return false;
  }
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
  at::DataPtr old_data = storage->set_data_ptr(std::move(new_data));
  const auto old_capacity = storage->nbytes();
  storage->set_nbytes(size_bytes);
  const auto copy_capacity = std::min(size_bytes, old_capacity);
  if (old_data != nullptr && copy_capacity > 0) {
    memcpy(storage->data(), old_data.get(), copy_capacity);
  }
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
    c10::optional<MemoryFormat> optional_memory_format) {
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

const Tensor& resize_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  // NOLINTNEXTLINE(bugprone-argument-comment)
  resize_impl_cpu_(self_, size, /*strides=*/c10::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  return self;
}

void resize_bytes_meta(StorageImpl* storage, c10::SymInt size_bytes) {
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
  storage->set_nbytes(size_bytes);
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
  } else if (new_size_bytes > storage.nbytes()) {
    resize_bytes_meta(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

inline TensorImpl* resize_impl_meta_(
    TensorImpl* self,
    c10::SymIntArrayRef size,
    at::OptionalSymIntArrayRef stride,
    bool resize_storage = true) {
  if (self->sym_sizes() == size && (!stride || self->sym_strides() == stride.value())) {
    return self;
  }

  const auto itemsize = self->dtype().itemsize();
  const auto storage_offset = self->sym_storage_offset();
  auto storage_size = c10::SymInt(1);
  std::vector<SymInt> strides;
  int64_t dim = size.size();
  if (!stride.has_value()) {
    strides.resize(dim);
    // TODO: Move this into TensorImpl
    // this is duped with empty_symint_meta
    if (dim > 0) {
      const auto last_idx = dim - 1;
      strides.at(last_idx) = 1;
      for (auto i = last_idx - 1; i >= 0; --i) {
        // TODO: max with 1
        strides.at(i) = strides.at(i+1) * size.at(i+1);
      }
    }
    stride = c10::make_optional<c10::SymIntArrayRef>(strides);
  }
  self->set_sizes_and_strides(size, *stride);
  storage_size = at::detail::computeStorageNbytes(
      size, *stride, itemsize, storage_offset);
  if (resize_storage) {
    maybe_resize_storage_meta(self, storage_size);
  }

  return self;
}

const Tensor& resize__symint(
    const Tensor& self,
    c10::SymIntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  TORCH_INTERNAL_ASSERT(!self.has_names())
  auto* self_ = self.unsafeGetTensorImpl();
  // NOLINTNEXTLINE(bugprone-argument-comment)
  resize_impl_meta_(self_, size, /*strides=*/c10::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    // NB: This doesn't actually work with symbolic shapes
    self_->empty_tensor_restride(memory_format);
  }
  return self;
}

} // namespace native
} // namespace at

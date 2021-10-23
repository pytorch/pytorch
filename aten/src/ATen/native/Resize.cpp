#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>

#include <c10/core/TensorOptions.h>

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
      "output shape ", shape, ".",
      "This behavior is deprecated, and in a future PyTorch release outputs ",
      "will not be resized unless they have zero elements. You can explicitly ",
      "reuse an out tensor t by resizing it, inplace, to zero elements with ",
      "t.resize_(0).");
  }
  return true;
}

static auto kFunctorchWrappedTensors = DispatchKeySet({
    DispatchKey::FuncTorchGradWrapper,
    DispatchKey::FuncTorchBatched,
    DispatchKey::FuncTorchPython});

static bool is_functorch_wrapped_tensor(const Tensor& tensor) {
  auto key_set = tensor.unsafeGetTensorImpl()->key_set();
  return !(key_set & kFunctorchWrappedTensors).empty();
}

bool resize_output(const Tensor& output, IntArrayRef shape, c10::optional<IntArrayRef> strides) {
  if (resize_output_check(output, shape)) {
    // TODO(#61485): functorch wrapped tensors should not go through the
    // fast path. This is a hack, longer term solutions are in the issue
    if (output.is_cpu() && !is_functorch_wrapped_tensor(output)) {
      at::native::resize_template<&resize_impl_tryreuse_<&maybe_resize_storage_cpu>>(
          output, shape, strides, c10::nullopt, true);
    } else {
      // We need to dispatch to different functions based on whether
      // `strides` is given or not.
      if (strides.has_value()) {
        output.resize_(shape, strides.value());
      } else {
        output.resize_(shape);
      }
    }
    return true;
  } else {
    return false;
  }
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

uint64_t select_storage_size_default(
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

uint64_t select_storage_size_tryreuse(
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

TensorImpl* resize_impl_cpu_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool resize_storage) {
  return resize_impl_template_<
      &maybe_resize_storage_cpu,
      &select_storage_size_default>(self, size, stride);
}

const Tensor& resize_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  resize_template<&resize_impl_cpu_>(self, size, c10::nullopt, optional_memory_format, true);
  return self;
}

const Tensor& resize_with_strides_(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef strides) {
  resize_template<&resize_impl_tryreuse_<&maybe_resize_storage_cpu>>(
      self, size, strides, c10::nullopt, true);
  return self;
}

} // namespace native
} // namespace at

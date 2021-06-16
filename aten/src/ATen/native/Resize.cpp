#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>
#include <torch/library.h>
#include <c10/core/TensorOptions.h>

namespace at { namespace native {

// Returns true if resize is necessary
bool resize_output_check(const Tensor& output, IntArrayRef shape) {
  // Tests for resizing of tensors with one more elements
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

bool resize_output(const Tensor& output, IntArrayRef shape) {
  if (resize_output_check(output, shape)) {
    // avoid a redispatch for cpu and cuda.
    // TODO: when resize_cuda_ is re-written to be unified with resize_,
    // we can provide the same benefit for cuda.
    if (output.is_cpu()) {
      at::native::resize_(output, shape);
    } else {
      output.resize_(shape);
    }
    return true;
  } else {
    return false;
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

} // namespace native
} // namespace at

#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>
#include <c10/core/TensorOptions.h>

namespace at { namespace native {

Tensor& resize_cpu_(Tensor& self, IntArrayRef size) {
#ifdef BUILD_NAMEDTENSOR
  if (self.has_names()) {
    return resize_named_tensor_(self, size);
  }
#endif
  auto* self_ = self.unsafeGetTensorImpl();
  resize_impl_cpu_(self_, size, /*strides=*/c10::nullopt);
  self_->maybe_zero_dim(size.size() == 0);
  return self;
}

// Call the sparse implementation in SparseTensor.cpp directly.
// A dynamic dispatch here is NOT necessary, so I didn't put
// this function in native_functions.yaml
Tensor& resize_as_sparse_(Tensor& self, const Tensor& src);


// TODO(VitalyFedyunin): Move it to HTML docs.
//
// Strides of the output tensor of `resize_as_` operator is defined by input
// tensor strides and the value of memory_format argument.
//
// If memory_format is not defined or equals to MemoryFormat::Contiguous
// (torch.contiguous_format) output tensor will have contiguous strides.
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
Tensor& resize_as_(
    Tensor& self,
    const Tensor& the_template,
    c10::optional<MemoryFormat> optional_memory_format) {
  if (self.is_sparse() && the_template.is_sparse()) {
    TORCH_CHECK(
        !optional_memory_format.has_value(),
        "Unsupported memory format for sparse tensor resize_as_ :",
        optional_memory_format.value());
    return native::resize_as_sparse_(self, the_template);
  }
  Tensor& result = self.resize_(the_template.sizes());
  if (optional_memory_format.has_value()) {
    auto memory_format = optional_memory_format.value();
    if (memory_format == MemoryFormat::Preserve) {
      memory_format = the_template.suggest_memory_format();
    }
    self.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  }
#ifdef BUILD_NAMEDTENSOR
  namedinference::propagate_names(result, the_template);
#endif
  return result;
}

} // namespace native
} // namespace at

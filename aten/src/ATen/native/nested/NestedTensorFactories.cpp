#include <ATen/ATen.h>
#include <ATen/native/nested/NestedTensorFactories.h>
#include <ATen/native/nested/NestedTensorUtils.h>

namespace at {
namespace native {

TensorOptions verify_empty_parameters(
    const at::Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");
  TensorOptions options = self.options().merge_in(options_).merge_memory_format(
      optional_memory_format);

  auto memory_format =
      options_.memory_format_opt().value_or(MemoryFormat::Preserve);
  TORCH_CHECK(
      memory_format == MemoryFormat::Preserve,
      "empty_like_nested only supports memory format Preserve, but got ",
      memory_format,
      " instead.");

  TORCH_CHECK(
      self.is_contiguous(),
      "empty_like only supports contiguous memory format for Nested Tensors");

  TORCH_CHECK(
      !(options.layout() != kStrided && optional_memory_format.has_value()),
      "memory format option is only supported by strided tensors");
  return options;
}

Tensor empty_like_nested(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto options = verify_empty_parameters(
      self, dtype, layout, device, pin_memory, optional_memory_format);
  auto self_nt = get_nested_tensor_impl(self);
  Tensor new_buffer = at::empty_like(self_nt->get_buffer(), options);
  auto nested_size = self_nt->get_nested_size_tensor().clone();
  auto nested_strides = self_nt->get_nested_stride_tensor().clone();
  auto offsets = std::vector<int64_t>(self_nt->get_storage_offsets());
  auto tensor = detail::make_tensor_base<NestedTensorImpl>(
      new_buffer, nested_size, nested_strides, std::move(offsets));
  return tensor;
}

// Take a Device that may not have device_index set (i.e., having it as -1
// representing the current device) and return the corresponding Device
// according to the actual device at the time of this function call.  No-op
// if the device_index is set.
static inline Device ensure_has_index(Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

Tensor _to_copy_nested(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !layout.has_value() || self.layout() == layout.value(),
      "to(options) doesn't support converting to a different layout, "
      "but got self.layout being ",
      self.layout(),
      " and options.layout set as ",
      layout.value());
  auto options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  if (options.has_device()) {
    options = options.device(ensure_has_index(options.device()));
  }
  // memory_format is handled separately due to MemoryFormat::Preserve logic
  options = self.options().merge_in(options).memory_format(c10::nullopt);
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);

  bool pin_out =
      (non_blocking && self.is_cuda() && options.device().is_cpu() &&
       (options.layout() == c10::kStrided));

  Tensor r;
  r = at::empty_like(self, dtype, layout, device, pin_out, memory_format);
  get_nested_tensor_impl(r)->get_buffer().copy_(
      get_nested_tensor_impl(self)->get_buffer(), non_blocking);
  return r;
}

Tensor& copy_nested_(Tensor& self, const Tensor& src, bool non_blocking) {
  const auto* nt_self = get_nested_tensor_impl(self);
  const auto* nt_src = get_nested_tensor_impl(src);
  TORCH_CHECK(
      at::equal(
          nt_self->get_nested_size_tensor(), nt_src->get_nested_size_tensor()),
      "copy_ only supports tensors that are the same size for Nested implementations");
  nt_self->get_buffer().copy_(nt_src->get_buffer(), non_blocking);
  return self;
}

} // namespace native
} // namespace at

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Optional.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace at {
namespace native {

// Take a Device that may not have device_index set (i.e., having it as -1
// representing the current device) and return the corresponding Device
// according to the actual device at the time of this function call.  No-op
// if the device_index is set.
static inline Device ensure_has_index(Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl = c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

static inline Tensor to_impl(const Tensor& self, const TensorOptions& options, bool non_blocking, bool copy) {
  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Preserve);

  if (self.dtype() == options.dtype() && self.layout() == options.layout() &&
      self.device() == options.device() && !copy &&
      (memory_format == MemoryFormat::Preserve ||
       self.suggest_memory_format() == memory_format)) {
    return self;
  }

  bool pin_out = (non_blocking && self.is_cuda() && options.device().is_cpu() &&
                  (options.layout() == c10::kStrided));

  if (memory_format == MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      // Copy all strides
      auto r = at::empty_strided(self.sizes(),
                                 self.strides(),
                                 options.memory_format(c10::nullopt).pinned_memory(pin_out));
      r.copy_(self, non_blocking);
      return r;
    } else {
      memory_format = self.suggest_memory_format();
    }
  }
  // See Note [Explicit nullopt MemoryFormat argument]
  auto r = at::empty(self.sizes(),
                     options.memory_format(memory_format).pinned_memory(pin_out),
                     c10::nullopt);
  r.copy_(self, non_blocking);
  return r;
}

Tensor to(
  const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
  bool non_blocking,
  bool copy,
  c10::optional<c10::MemoryFormat> optional_memory_format
) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  TORCH_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");
  auto options = options_.merge_memory_format(optional_memory_format);

  TORCH_CHECK(options.requires_grad_opt() == c10::nullopt,
           "to(options) expects unset requires_grad flag, but got "
           "options.requires_grad set as ", options.requires_grad());

  TORCH_CHECK(!options.has_layout() || self.layout() == options.layout(),
           "to(options) doesn't support converting to a different layout, "
           "but got self.layout being ", self.layout(),
           " and options.layout set as ", options.layout());

  if (options.has_device()) {
    options = options.device(ensure_has_index(options.device()));
  }
  auto specified_options = self.options().merge_in(options);
  return to_impl(self, specified_options, non_blocking, copy);
}

Tensor to(const Tensor& self, Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  device = ensure_has_index(device);
  return to_impl(
      self,
      self.options().device(device).dtype(dtype).memory_format(optional_memory_format),
      non_blocking,
      copy);
}

Tensor to(const Tensor& self, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  return to_impl(
      self, self.options().dtype(dtype).memory_format(optional_memory_format), non_blocking, copy);
}

Tensor to(const Tensor& self, const Tensor& other, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto options = other.options();
  return to_impl(self, options.memory_format(optional_memory_format), non_blocking, copy);
}

Tensor to_dense_backward(const Tensor& grad, const Tensor& input_) {
  AT_ASSERT(input_.layout() != c10::kStrided);
  if (input_.layout() == c10::kSparse) {
    auto input = input_.coalesce();
    return grad.sparse_mask(input);
  } else if (input_.layout() == c10::kMkldnn) {
    return grad.to_mkldnn(input_.scalar_type());
  } else {
    AT_ERROR("Unsupported input layout: ", input_.layout());
  }
}

Tensor to_mkldnn_backward(const Tensor& grad, const Tensor& input_) {
  AT_ASSERT(input_.layout() == c10::kStrided);
  return grad.to_dense(input_.scalar_type());
}

Tensor view_dtype(const Tensor& self, ScalarType dtype) {
  if (self.scalar_type() == dtype) {
    return self;
  }
  auto type_meta = c10::scalarTypeToTypeMeta(dtype);
  TORCH_CHECK(self.element_size() == type_meta.itemsize(),
    "Viewing a tensor as a new dtype with a different number of bytes per element is not supported.");
  Storage storage = self.storage();
  auto new_tensor = detail::make_tensor<TensorImpl>(
      std::move(storage), self.key_set(), type_meta);
  auto* impl = new_tensor.unsafeGetTensorImpl();
  impl->set_storage_offset(self.storage_offset());
  impl->set_sizes_and_strides(self.sizes(), self.strides());
  return new_tensor;
}

}} // namespace at::native

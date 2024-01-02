#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorFactories.h>
#include <ATen/native/nested/NestedTensorUtils.h>

namespace at {
namespace native {

static TensorOptions verify_empty_parameters(
    const at::Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ = TensorOptions()
                               .dtype(dtype)
                               .layout(layout)
                               .device(device)
                               .pinned_memory(pin_memory)
                               .memory_format(optional_memory_format);

  TensorOptions options = self.options().merge_in(options_);
  auto memory_format =
      options_.memory_format_opt().value_or(MemoryFormat::Preserve);
  TORCH_CHECK(
      memory_format == MemoryFormat::Preserve || memory_format == MemoryFormat::Contiguous,
      "empty_like_nested only supports memory format Preserve or Contiguous, but got ",
      memory_format,
      " instead.");

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
  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Preserve);
  if (memory_format == MemoryFormat::Contiguous) {
    auto nested_size = self_nt->get_nested_sizes().clone();
    int64_t buffer_size = get_numel_from_nested_size_tensor(nested_size);
    Tensor new_buffer = at::empty({buffer_size}, options);
    auto tensor = wrap_buffer(new_buffer, nested_size);
    return tensor;
  }
  // The fall through path must be Preserve
  TORCH_CHECK(
      memory_format == MemoryFormat::Preserve,
      "memory format option is only supported by strided tensors");
  // Since we clone sizes, strides, and offsets it should be safe to use
  // get_unsafe_storage_as_tensor for the call to empty_like.
  Tensor new_buffer =
      at::empty_like(self_nt->get_unsafe_storage_as_tensor(), options);
  auto nested_size = self_nt->get_nested_sizes().clone();
  auto nested_strides = self_nt->get_nested_strides().clone();
  auto offsets = self_nt->get_storage_offsets().clone();
  auto tensor = wrap_buffer(new_buffer, nested_size, nested_strides, offsets);
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
          nt_self->get_nested_sizes(), nt_src->get_nested_sizes()),
      "copy_ only supports tensors that are the same size for Nested implementations");
  nt_self->get_buffer().copy_(nt_src->get_buffer(), non_blocking);
  return self;
}


Tensor clone_nested(
    const Tensor& self,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(c10::MemoryFormat::Preserve);
  auto self_ptr = get_nested_tensor_impl(self);
  if (memory_format == c10::MemoryFormat::Preserve ||
  (memory_format == c10::MemoryFormat::Contiguous && self.is_contiguous())) {
    const Tensor& buffer = self_ptr->get_unsafe_storage_as_tensor(),
        sizemat = self_ptr->get_nested_sizes(),
        stridemat = self_ptr->get_nested_strides();
    const auto& offsets = self_ptr->get_storage_offsets();
    // TODO: The size and the stride do not necessarily need to be cloned,
    //       but it is more conservative.
    //       This is something we could revisit once we land a more
    //       efficient implementation of nested_sizes_ and nested_strides.
    return wrap_buffer(buffer.clone(), sizemat.clone(), stridemat.clone(), offsets.clone());
  }
  // actually, memory format is contiguous and self is noncontiguous
  else if (memory_format == c10::MemoryFormat::Contiguous) {
    const Tensor& self_buffer = self_ptr->get_unsafe_storage_as_tensor(),
        sizemat = self_ptr->get_nested_sizes();
    Tensor output_buffer = at::empty(self.numel(), self_buffer.options());
    Tensor output = wrap_buffer(output_buffer, sizemat);
    std::vector<Tensor> self_unbind = self.unbind(),
        output_unbind = output.unbind();
    for (const int64_t i: c10::irange(self_ptr->size(0))) {
      output_unbind[i].copy_(self_unbind[i]);
    }
    return output;
  } else {
    TORCH_CHECK(
        false,
        "Nested tensor clone supports Preserve and Contiguous memory formats, called clone with memory format: ",
        memory_format);
  }
}

std::vector<at::Tensor> NestedTensor_unbind(
    const at::Tensor& self,
    int64_t dim) {
  TORCH_CHECK(
      dim == 0,
      "NestedTensor can only be unbound along dimension 0 ",
      "got dimension ",
      dim,
      " instead.");
  auto self_ptr = get_nested_tensor_impl(self);
  int64_t ntensors = self_ptr->size(0);
  std::vector<at::Tensor> result_tensors(ntensors);
  if (ntensors == 0) {
    return result_tensors;
  }
  // This returns a differentiable view of self as a regular tensor
  auto buffer = self.values();
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
      strides = NestedTensor_get_strides(self_ptr);
  int64_t *offsets_ptr = self_ptr->get_storage_offsets().data_ptr<int64_t>();
  for (const int64_t i: c10::irange(ntensors)){
    result_tensors[i] = buffer.as_strided(sizes[i], strides[i], offsets_ptr[i]);
  }
  return result_tensors;
}

Tensor narrow_nested_symint(const at::Tensor& self, int64_t dim, SymInt start, SymInt length) {
  TORCH_CHECK(dim == 0, "narrow(): only dim=0 supported for nested tensors, but got: ", dim);
  TORCH_SYM_CHECK(length.sym_ge(0), "narrow(): length must be non-negative");
  auto cur_size = self.sym_size(dim);
  TORCH_CHECK_INDEX(
      ((-cur_size).sym_le(start).sym_and(start.sym_le(cur_size))).expect_true(__FILE__, __LINE__),
      "start out of range (expected to be in range of [", -cur_size, ", ", cur_size, "], but got ",
      start, ")");
  if (start < 0) {
    start = start + cur_size;
  }
  TORCH_SYM_CHECK(start.sym_le(cur_size - length),
      "start (", start, ") + length (", length, ") exceeds dimension size (", cur_size, ").");
  auto *nt_impl = get_nested_tensor_impl(self);
  TORCH_CHECK(self.is_contiguous(), "narrow(): only contiguous nested tensors supported");
  auto buffer = nt_impl->get_unsafe_storage_as_tensor();
  auto nested_sizes = nt_impl->get_nested_sizes();
  auto nested_strides = nt_impl->get_nested_strides();
  auto storage_offsets = nt_impl->get_storage_offsets();
  auto storage_offsets_ptr = storage_offsets.data_ptr<int64_t>();

  auto start_int = start.expect_int();
  auto length_int = length.expect_int();
  auto buffer_offset = storage_offsets_ptr[start_int];

  nested_sizes = nested_sizes.narrow(0, start_int, length_int);
  nested_strides = nested_strides.narrow(0, start_int, length_int);
  storage_offsets = storage_offsets.narrow(0, start_int, length_int);

  return at::detail::make_tensor<NestedTensorImpl>(
      c10::TensorImpl::VIEW,
      buffer.narrow(0, buffer_offset, buffer.numel() - buffer_offset),
      nested_sizes,
      nested_strides,
      storage_offsets);
}

} // namespace native
} // namespace at

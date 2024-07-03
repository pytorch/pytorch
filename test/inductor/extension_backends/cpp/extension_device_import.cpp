#include <torch/extension.h>
#include <c10/core/CPUAllocator.h>
#include <ATen/ops/eq_ops.h>
#include <ATen/ops/fill_ops.h>
#include <ATen/ops/view_ops.h>
#include <ATen/ops/zero_ops.h>
#include <ATen/ops/_local_scalar_dense_ops.h>

constexpr c10::DispatchKeySet dense_cpu = c10::DispatchKeySet(
    c10::BackendComponent::CPUBit).add(c10::DispatchKey::Dense);

at::Tensor custom_as_strided(const at::Tensor& self, at::IntArrayRef size,
                             at::IntArrayRef stride, at::optional<int64_t> storage_offset_) {
    return at::native::as_strided_tensorimpl(self, size, stride, storage_offset_);
}

at::Tensor custom_empty_memory_format(at::IntArrayRef size,
                                           std::optional<at::ScalarType> dtype,
                                           std::optional<at::Layout> layout,
                                           std::optional<at::Device> device,
                                           std::optional<bool> pin_memory,
                                           std::optional<at::MemoryFormat> memory_format) {
    return at::detail::empty_generic(size,
                                     c10::GetDefaultCPUAllocator(),
                                     c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                     c10::dtype_or_default(dtype),
                                     memory_format);
}


at::Tensor &custom_eq_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self,
                                 const at::Tensor & other, at::Tensor &out) {
    return at::_ops::eq_Tensor_out::redispatch(dense_cpu, self, other, out);
}

at::Tensor &custom_fill_(at::Tensor& self, const at::Scalar& value) {
    return at::_ops::fill__Scalar::redispatch(dense_cpu, self, value);
}

at::Tensor custom_view(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size) {
    return at::_ops::view::redispatch(dense_cpu, self, size);
}

at::Tensor &custom_zero_(at::Tensor &self){
    return at::_ops::zero_::redispatch(dense_cpu, self);
}

at::Scalar custom__local_scalar_dense_(const at::Tensor& self) {
    return at::_ops::_local_scalar_dense::redispatch(dense_cpu, self);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("as_strided", custom_as_strided);
    m.impl("empty.memory_format", &custom_empty_memory_format);
    m.impl("eq.Tensor_out", &custom_eq_Tensor_out);
    m.impl("fill_.Scalar", &custom_fill_);
    m.impl("view", &custom_view);
    m.impl("zero_", &custom_zero_);
    m.impl("_local_scalar_dense", &custom__local_scalar_dense_);
}

c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_custom_device", &get_custom_device, "get custom device object");
}

#include <c10/core/CPUAllocator.h>
#include <torch/extension.h>

at::Tensor custom_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                                std::optional<at::ScalarType> dtype_opt,
                                std::optional<at::Layout> layout_opt,
                                std::optional<at::Device> device_opt,
                                std::optional<bool> pin_memory_opt) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return at::detail::empty_strided_generic(size, stride, c10::GetDefaultCPUAllocator(), private_use_ks, dtype);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("_empty_strided_extension_device_import",
            &custom_empty_strided,
             "mimic _empty_strided_cpu for test");
}

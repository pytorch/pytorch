// ${generated_comment}

#include <ATen/Functions.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/TypeDefault.h>
#include <ATen/CPUType.h>
#include <ATen/QuantizedCPUType.h>
#ifdef USE_VULKAN
#include <ATen/VulkanType.h>
#endif

namespace at {

namespace {
  // TODO The maybe_unwrap_optional_tensor is only needed because our at::native::xxx functions
  // still take "Tensor" instead of "optional<Tensor>", so we need CPUType, TypeDefault, ...
  // to do the same. Once at::native::xxx are converted, we can remove use_optional_tensor
  // and use the use_optional_tensor=True behavior always.
  template<class T, std::enable_if_t<!std::is_same<c10::optional<at::Tensor>, std::decay_t<T>>::value, int> = 0>
  decltype(auto) maybe_unwrap_optional_tensor(T&& arg) {
    return std::forward<T>(arg);
  }
  Tensor maybe_unwrap_optional_tensor(const c10::optional<at::Tensor>& arg) {
    if (arg.has_value()) {
      return *arg;
    } else {
      return Tensor();
    }
  }
}

using native::tensor;

${function_definitions}

}

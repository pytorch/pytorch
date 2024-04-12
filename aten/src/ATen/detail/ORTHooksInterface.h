#pragma once

#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

constexpr const char* ORT_HELP =
  " You need to 'import torch_ort' to use the 'ort' device in PyTorch. "
  "The 'torch_ort' module is provided by the ONNX Runtime itself "
  "(https://onnxruntime.ai).";

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {

struct TORCH_API ORTHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~ORTHooksInterface() = default;

  virtual std::string showConfig() const {
    TORCH_CHECK(false, "Cannot query detailed ORT version information.", ORT_HELP);
  }
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct TORCH_API ORTHooksArgs {};

TORCH_DECLARE_REGISTRY(ORTHooksRegistry, ORTHooksInterface, ORTHooksArgs);
#define REGISTER_ORT_HOOKS(clsname) \
  C10_REGISTER_CLASS(ORTHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const ORTHooksInterface& getORTHooks();
} // namespace detail

} // namespace at

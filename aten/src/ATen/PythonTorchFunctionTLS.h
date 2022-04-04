#pragma once

#include <c10/macros/Macros.h>

namespace at {
namespace impl {

struct TORCH_API PythonTorchFunctionTLS {
  static void set_disabled(bool);
  static bool is_disabled();

  static void set_state(const PythonTorchFunctionTLS& state);
  static const PythonTorchFunctionTLS& get_state();

private:
  bool disabled_;
};

} // namespace impl
} // namespace at

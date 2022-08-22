#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Macros.h>

namespace at {
namespace impl {

struct TORCH_API PythonTorchFunctionTLS {
  static void set_disabled(bool);
  static bool is_disabled();

  static void set_mode(std::shared_ptr<c10::SafePyObject>);
  static const std::shared_ptr<c10::SafePyObject>& get_mode();
  static void swap_mode(std::shared_ptr<c10::SafePyObject>&);

  static bool exchange_skip_next(bool);
  static bool peek_skip_next();

  static void set_state(const PythonTorchFunctionTLS& state);
  static const PythonTorchFunctionTLS& get_state();

 private:
  bool disabled_;
  bool skip_next_;
  std::shared_ptr<c10::SafePyObject> mode_;
};

} // namespace impl
} // namespace at

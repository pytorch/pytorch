#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Macros.h>

namespace at {
namespace impl {

struct TORCH_API PythonTorchFunctionTLS {
  static void set_disabled(bool);
  static bool is_disabled();

  static void push_onto_stack(std::shared_ptr<SafePyObject> mode);
  static const std::shared_ptr<SafePyObject> pop_stack();
  static const std::shared_ptr<SafePyObject>& get_stack_at(int64_t idx);
  static int64_t stack_len();

  static const PythonTorchFunctionTLS& get_state();
  static void set_state(const PythonTorchFunctionTLS& state);

 private:
  // The mode TLS is split into
  //   - disabled_, which says whether or not to disable all torch function
  //   modes
  //   - stack_, which is a vector of modes representing the stack of user
  //   defined modes
  bool disabled_;
  std::vector<std::shared_ptr<c10::SafePyObject>> stack_;
};

TORCH_API bool torch_function_mode_enabled();

} // namespace impl
} // namespace at

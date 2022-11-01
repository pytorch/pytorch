#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Macros.h>

namespace at {
namespace impl {

struct TORCH_API PythonTorchFunctionTLS {
  static void set_disable_subclass(bool);
  static void set_disable_all(bool);
  static bool is_disable_subclass();
  static bool is_disable_all();

  static void push_onto_stack(std::shared_ptr<SafePyObject> mode);
  static const std::shared_ptr<SafePyObject> pop_stack();
  static const std::shared_ptr<SafePyObject>& get_stack_at(int64_t idx);
  static int64_t stack_len();

  static const PythonTorchFunctionTLS& get_state();
  static void set_state(const PythonTorchFunctionTLS& state);

 private:
  // The mode TLS is split into
  //   - disable_subclass_, which says whether or not to disable torch function
  //     on subclasses
  //   - disable_all_, which says whether or not to disable torch function on
  //     both subclasses and modes. If this is true, it takes precedence over
  //     disable_subclass
  //   - stack_, which is a vector of modes representing the stack of user
  //   defined modes
  bool disable_subclass_ = false;
  bool disable_all_ = false;
  std::vector<std::shared_ptr<c10::SafePyObject>> stack_;
};

TORCH_API bool torch_function_mode_enabled();

} // namespace impl
} // namespace at

#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

namespace c10 {
namespace impl {

struct C10_API TorchDispatchModeTLS {
  static void set_mode(std::shared_ptr<SafePyObject> mode);
  static const std::shared_ptr<SafePyObject>& get_mode();
  static void reset_mode();
  static void swap_mode(std::shared_ptr<SafePyObject>& mode);

  static void push_onto_stack(std::shared_ptr<SafePyObject> mode);
  static const std::shared_ptr<SafePyObject> pop_stack();
  static const std::shared_ptr<SafePyObject>& get_stack_at(int64_t idx);
  static int64_t stack_len();

  static const TorchDispatchModeTLS& get_state();
  static void set_state(const TorchDispatchModeTLS& state);

 private:
  // The mode TLS is split into
  //   - mode_, which is the C++ mode, that can only be the mode handling mode
  //   or null
  //   - stack_, which is a vector of modes representing the stack of user
  //   defined modes
  std::shared_ptr<c10::SafePyObject> mode_;
  std::vector<std::shared_ptr<c10::SafePyObject>> stack_;
};

C10_API bool dispatch_mode_enabled();

} // namespace impl
} // namespace c10

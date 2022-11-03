#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>

namespace c10 {
namespace impl {

struct C10_API PythonDispatcherTLS {
  static void set_state(PythonDispatcherTLS state);
  static PythonDispatcherTLS get_state();
  static void user_set_state(PyInterpreter* interpreter);
  static PyInterpreter* get_interpreter();
  static void set_interpreter(PyInterpreter* state);

  static void push_onto_pre_stack(
      std::shared_ptr<SafePyObject> mode,
      PyInterpreter* interpreter);
  static const std::shared_ptr<SafePyObject> pop_pre_stack();
  static const std::shared_ptr<SafePyObject>& get_pre_stack_at(int64_t idx);
  static int64_t pre_stack_len();

 private:
  std::vector<std::shared_ptr<c10::SafePyObject>> pre_dispatch_stack_;
  PyInterpreter* interpreter_;
  bool user_activated = false;
};

struct C10_API DisablePythonDispatcher {
  DisablePythonDispatcher() : old_(PythonDispatcherTLS::get_interpreter()) {
    PythonDispatcherTLS::set_interpreter(nullptr);
  }
  ~DisablePythonDispatcher() {
    PythonDispatcherTLS::set_interpreter(old_);
  }
  PyInterpreter* old_;
};

} // namespace impl
} // namespace c10

#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <c10/macros/Export.h>

namespace c10::impl {

struct C10_API PythonDispatcherTLS {
  static void set_state(PyInterpreter* state);
  static PyInterpreter* get_state();
  static void reset_state();
};

struct C10_API DisablePythonDispatcher {
  DisablePythonDispatcher() : old_(PythonDispatcherTLS::get_state()) {
    PythonDispatcherTLS::set_state({});
  }

  DisablePythonDispatcher(DisablePythonDispatcher&& other) = delete;
  DisablePythonDispatcher(const DisablePythonDispatcher&) = delete;
  DisablePythonDispatcher& operator=(const DisablePythonDispatcher&) = delete;
  DisablePythonDispatcher& operator=(DisablePythonDispatcher&&) = delete;
  ~DisablePythonDispatcher() {
    PythonDispatcherTLS::set_state(old_);
  }
  PyInterpreter* old_;
};

} // namespace c10::impl

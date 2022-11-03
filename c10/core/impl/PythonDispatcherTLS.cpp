#include <c10/core/DispatchKeySet.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <iostream>

namespace c10 {
namespace impl {

thread_local PythonDispatcherTLS pythonDispatcherState;

void PythonDispatcherTLS::set_state(PythonDispatcherTLS state) {
  pythonDispatcherState = state;
  if (pythonDispatcherState.interpreter_) {
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonDispatcher, true);
  } else {
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonDispatcher, false);
  }
}

void PythonDispatcherTLS::user_set_state(PyInterpreter* interpreter) {
  if (interpreter) {
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonDispatcher, true);
    pythonDispatcherState.user_activated = true;
    pythonDispatcherState.interpreter_ = interpreter;
  } else if (pre_stack_len() > 0) {
    pythonDispatcherState.user_activated = false;
  } else {
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonDispatcher, false);
    pythonDispatcherState.user_activated = false;
    pythonDispatcherState.interpreter_ = nullptr;
  }
}

PyInterpreter* PythonDispatcherTLS::get_interpreter() {
  return pythonDispatcherState.interpreter_;
}

PythonDispatcherTLS PythonDispatcherTLS::get_state() {
  return pythonDispatcherState;
}

int64_t PythonDispatcherTLS::pre_stack_len() {
  return pythonDispatcherState.pre_dispatch_stack_.size();
}

void PythonDispatcherTLS::set_interpreter(PyInterpreter* interpreter) {
  if (interpreter && PythonDispatcherTLS::pre_stack_len() > 0 && pythonDispatcherState.user_activated) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonDispatcher, true);
    pythonDispatcherState.interpreter_ = interpreter;
  } else {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonDispatcher, false);
    pythonDispatcherState.interpreter_ = nullptr;
  }
}

const std::shared_ptr<SafePyObject>& PythonDispatcherTLS::get_pre_stack_at(
    int64_t idx) {
  TORCH_CHECK(
      idx < pre_stack_len(), "Tried to get stack at idx that's too big");
  return pythonDispatcherState.pre_dispatch_stack_[idx];
}

const std::shared_ptr<SafePyObject> PythonDispatcherTLS::pop_pre_stack() {
  TORCH_CHECK(
      pythonDispatcherState.pre_dispatch_stack_.size() > 0,
      "trying to pop from empty mode stack");
  const std::shared_ptr<SafePyObject> out =
      pythonDispatcherState.pre_dispatch_stack_.back();
  pythonDispatcherState.pre_dispatch_stack_.pop_back();
  if (pythonDispatcherState.pre_dispatch_stack_.size() == 0 &&
      !pythonDispatcherState.user_activated) {
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonDispatcher, false);
    pythonDispatcherState.interpreter_ = nullptr;
  }
  return out;
}

void PythonDispatcherTLS::push_onto_pre_stack(
    std::shared_ptr<SafePyObject> mode,
    PyInterpreter* interpreter) {
  if (pythonDispatcherState.pre_dispatch_stack_.size() == 0 &&
      !pythonDispatcherState.interpreter_) {
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonDispatcher, true);
    pythonDispatcherState.interpreter_ = interpreter;
  }
  pythonDispatcherState.pre_dispatch_stack_.push_back(std::move(mode));
}

} // namespace impl
} // namespace c10

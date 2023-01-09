#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace torch {
namespace torch_dispatch_mode {

struct StashTorchDispatchModeGuard {
 public:
  StashTorchDispatchModeGuard() {
    saved_mode_ = c10::impl::TorchDispatchModeTLS::pop_stack();
  }

  ~StashTorchDispatchModeGuard() {
    // since we're in the destructor, there might be active exceptions.
    // This temporarily removes them in order to update the state of the mode
    // before putting it back on the stack

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    c10::impl::TorchDispatchModeTLS::push_onto_stack(std::move(saved_mode_));
    PyErr_Restore(type, value, traceback);
  }

  const std::shared_ptr<c10::SafePyObject>& get_cur_mode() {
    return saved_mode_;
  }

 private:
  std::shared_ptr<c10::SafePyObject> saved_mode_;
};

struct StashTorchDispatchStackGuard {
 public:
  StashTorchDispatchStackGuard() {
    const auto old = c10::impl::TorchDispatchModeTLS::get_state();
    c10::impl::TorchDispatchModeTLS::set_state(saved_state_);
    saved_state_ = std::move(old);
  }

  ~StashTorchDispatchStackGuard() {
    // since we're in the destructor, there might be active exceptions.
    // This temporarily removes them in order to update the state of modes
    // on the stack before putting them back

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    c10::impl::TorchDispatchModeTLS::set_state(std::move(saved_state_));
    PyErr_Restore(type, value, traceback);
  }

 private:
  c10::impl::TorchDispatchModeTLS saved_state_;
};

} // namespace torch_dispatch_mode
} // namespace torch

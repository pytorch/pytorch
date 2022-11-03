#pragma once

#include <c10/core/impl/PythonDispatcherTLS.h>

namespace torch {
namespace torch_pre_dispatch_mode {

struct StashTorchPreDispatchModeGuard {
 public:
  StashTorchPreDispatchModeGuard() {
    interpreter_ = c10::impl::PythonDispatcherTLS::get_interpreter();
    saved_mode_ = c10::impl::PythonDispatcherTLS::pop_pre_stack();
  }

  ~StashTorchPreDispatchModeGuard() {
    c10::impl::PythonDispatcherTLS::push_onto_pre_stack(
        std::move(saved_mode_), interpreter_);
  }

  const std::shared_ptr<c10::SafePyObject>& get_cur_mode() {
    return saved_mode_;
  }

 private:
  c10::impl::PyInterpreter* interpreter_;
  std::shared_ptr<at::SafePyObject> saved_mode_;
};

} // namespace torch_pre_dispatch_mode
} // namespace torch

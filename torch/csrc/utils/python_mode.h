#pragma once

#include <ATen/core/PythonModeTLS.h>

namespace torch {
namespace python_dispatch {

struct StashPythonModeGuard {
public:
  StashPythonModeGuard() {
    saved_ = at::impl::PythonModeTLS::get_state();
    at::impl::PythonModeTLS::set_state(nullptr);
  }

  ~StashPythonModeGuard() {
    at::impl::PythonModeTLS::set_state(saved_);
  }
private:
  std::shared_ptr<at::SafePyObject> saved_;
};

} // namespace python_dispatch
} // namespace torch

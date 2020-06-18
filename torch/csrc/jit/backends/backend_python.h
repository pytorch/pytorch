#pragma once

#include <ATen/core/stack.h>

namespace torch {
namespace jit {
namespace detail {

using BackendRegistrationCallback = std::function<void(const std::string&)>;

// Add a function \p callback that should be invoked every time a backend is
// registered. The name of the backend is passed to the callback. This is
// primarily used for creating Python bindings for lowering to backends from
// Python.
void addBackendRegistrationCallback(BackendRegistrationCallback callback);
} // namespace detail
} // namespace jit
} // namespace torch

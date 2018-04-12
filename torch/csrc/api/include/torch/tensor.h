#pragma once

#include <torch/csrc/autograd/variable.h>

namespace at {
struct Type;
enum class Backend;
enum class ScalarType;
} // namespace at

namespace torch {
/// There will only be gradient recording tensors in the frontend API.
using Tensor = autograd::Variable;

/// Returns a `Type` object for the given backend (e.g. `at::kCPU`) and
/// `ScalarType` (e.g. `at::kDouble`).
at::Type& getType(at::Backend backend, at::ScalarType type);

/// Returns a `Type` object for the CPU backend and the given `ScalarType`
/// (e.g. `at::kDouble`). Equivalent to `getType(kCPU, type)`.
at::Type& CPU(at::ScalarType type);

/// Returns a `Type` object for the CUDA backend and the given `ScalarType`
/// (e.g. `at::kDouble`). Equivalent to `getType(kCUDA, type)`.
at::Type& CUDA(at::ScalarType type);
} // namespace torch

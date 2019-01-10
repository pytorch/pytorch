#pragma once

#include <Python.h>

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/THP_export.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {

// NOTE: This API is currently highly experimental and may change drastically
// in the near future.

/// Returns a `Type` object for the given backend (e.g. `at::kCPU`) and
/// `ScalarType` (e.g. `at::kDouble`).
THP_CLASS at::Type& getType(at::Backend backend, at::ScalarType type);

/// Returns a `Type` object for the CPU backend and the given `ScalarType`
/// (e.g. `at::kDouble`). Equivalent to `getType(kCPU, type)`.
THP_CLASS at::Type& CPU(at::ScalarType type);

/// Returns a `Type` object for the CUDA backend and the given `ScalarType`
/// (e.g. `at::kDouble`). Equivalent to `getType(kCUDA, type)`.
THP_CLASS at::Type& CUDA(at::ScalarType type);

/// Sets the `requires_grad` property of the given `Tensor`.
THP_CLASS void set_requires_grad(at::Tensor& tensor, bool requires_grad) noexcept;

/// Returns the `requires_grad` of the given `Tensor`.
THP_CLASS bool requires_grad(const at::Tensor& tensor) noexcept;
} // namespace torch

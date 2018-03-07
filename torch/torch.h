#pragma once

#include <Python.h>

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
/// Returns a `Type` object for the given backend (e.g. `at::kCPU`) and
/// `ScalarType` (e.g. `at::kDouble`).
autograd::VariableType getType(at::Backend backend, at::ScalarType type);

/// Returns a `Type` object for the CPU backend and the given `ScalarType`
/// (e.g. `at::kDouble`). Equivalent to `getType(kCPU, type)`.
autograd::VariableType CPU(at::ScalarType type);

/// Returns a `Type` object for the CUDA backend and the given `ScalarType`
/// (e.g. `at::kDouble`). Equivalent to `getType(kCUDA, type)`.
autograd::VariableType CUDA(at::ScalarType type);

/// Sets the `requires_grad` property of the given `Tensor`.
void set_requires_grad(at::Tensor& tensor, bool requires_grad) noexcept;

/// Returns the `requires_grad` of the given `Tensor`.
bool requires_grad(const at::Tensor& tensor) noexcept;
} // namespace torch

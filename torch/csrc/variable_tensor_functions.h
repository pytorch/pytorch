#pragma once

#include <ATen/ATen.h>
#include <ATen/core/Deprecated.h>
#include <torch/csrc/THP_export.h>

namespace torch {

// NOTE: This API is currently highly experimental and may change drastically
// in the near future.

// These functions provide a small wrapper around aten ensuring
// that we create tensors with type Variable rather than raw tensors
// when we create new tensors. We also provide a few accessors like
// requires_grad that make it easier to get to varible information when we have
// a at::Tensor

/// Returns a `TypeExtendedInterface` object for the given backend (e.g.
/// `at::kCPU`) and `ScalarType` (e.g. `at::kDouble`).
/// TODO: Eliminate this function as much as possible
AT_DEPRECATED(THP_CLASS at::TypeExtendedInterface& getVariableType(
    at::Backend backend,
    at::ScalarType type));

/// Returns a `TypeExtendedInterface` object for the CPU backend and the given
/// `ScalarType` (e.g. `at::kDouble`). Equivalent to `getVariableType(kCPU,
/// type)`.
/// TODO: Eliminate this function as much as possible
AT_DEPRECATED(THP_CLASS at::TypeExtendedInterface& CPU(at::ScalarType type));

/// Returns a `TypeExtendedInterface` object for the CUDA backend and the given
/// `ScalarType` (e.g. `at::kDouble`). Equivalent to `getVariableType(kCUDA,
/// type)`.
/// TODO: Eliminate this function as much as possible
AT_DEPRECATED(THP_CLASS at::TypeExtendedInterface& CUDA(at::ScalarType type));

/// Sets the `requires_grad` property of the given `Tensor`.
AT_DEPRECATED(THP_CLASS void set_requires_grad(
    at::Tensor& tensor,
    bool requires_grad) noexcept);

/// Returns the `requires_grad` of the given `Tensor`.
AT_DEPRECATED(THP_CLASS bool requires_grad(const at::Tensor& tensor) noexcept);

} // namespace torch

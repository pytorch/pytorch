#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/autograd/variable.h>

namespace torch {
// TODO: Rename to `Tensor`.
using Variable = autograd::Variable;

/// Special "raw data" dtype.
constexpr auto kByte = at::kByte;

/// Fixed width dtypes.
constexpr auto kInt8 = at::kChar;
constexpr auto kInt16 = at::kShort;
constexpr auto kInt32 = at::kInt;
constexpr auto kInt64 = at::kLong;
constexpr auto kFloat32 = at::kFloat;
constexpr auto kFloat64 = at::kDouble;

/// Rust-style short dtypes.
constexpr auto kI8 = at::kChar;
constexpr auto kI16 = at::kShort;
constexpr auto kI32 = at::kInt;
constexpr auto kI64 = at::kLong;
constexpr auto kF32 = at::kFloat;
constexpr auto kF64 = at::kDouble;

} // namespace torch

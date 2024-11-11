#pragma once

#include <ATen/ATen.h>

#include <optional>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/variable.h>

// TODO: These don't really belong here but torchvision builds in CI need them
// Remove once the torchvision version being compiled in CI is updated
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace torch {

// NOTE [ Exposing declarations in `at::` to `torch::` ]
//
// The following line `using namespace at;` is responsible for exposing all
// declarations in `at::` namespace to `torch::` namespace.
//
// According to the rules laid out in
// https://en.cppreference.com/w/cpp/language/qualified_lookup, section
// "Namespace members":
// ```
// Qualified lookup within the scope of a namespace N first considers all
// declarations that are located in N and all declarations that are located in
// the inline namespace members of N (and, transitively, in their inline
// namespace members). If there are no declarations in that set then it
// considers declarations in all namespaces named by using-directives found in N
// and in all transitive inline namespace members of N.
// ```
//
// This means that if both `at::` and `torch::` namespaces have a function with
// the same signature (e.g. both `at::func()` and `torch::func()` exist), after
// `namespace torch { using namespace at; }`, when we call `torch::func()`, the
// `func()` function defined in `torch::` namespace will always be called, and
// the `func()` function defined in `at::` namespace is always hidden.
using namespace at; // NOLINT

using std::nullopt; // NOLINT
using std::optional; // NOLINT

using Dtype = at::ScalarType;

/// Fixed width dtypes.
constexpr auto kUInt8 = at::kByte;
constexpr auto kInt8 = at::kChar;
constexpr auto kInt16 = at::kShort;
constexpr auto kInt32 = at::kInt;
constexpr auto kInt64 = at::kLong;
constexpr auto kUInt16 = at::kUInt16;
constexpr auto kUInt32 = at::kUInt32;
constexpr auto kUInt64 = at::kUInt64;
constexpr auto kFloat16 = at::kHalf;
constexpr auto kFloat32 = at::kFloat;
constexpr auto kFloat64 = at::kDouble;

/// Rust-style short dtypes.
constexpr auto kU8 = kUInt8;
constexpr auto kU16 = kUInt16;
constexpr auto kU32 = kUInt32;
constexpr auto kU64 = kUInt64;
constexpr auto kI8 = kInt8;
constexpr auto kI16 = kInt16;
constexpr auto kI32 = kInt32;
constexpr auto kI64 = kInt64;
constexpr auto kF16 = kFloat16;
constexpr auto kF32 = kFloat32;
constexpr auto kF64 = kFloat64;
} // namespace torch

// Copyright 2004-present Facebook. All Rights Reserved.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/FunctionalInverses.h>
#include <ATen/ScalarOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_test_ambiguous_defaults_native.h>
#include <ATen/ops/_test_autograd_multiple_dispatch_native.h>
#include <ATen/ops/_test_autograd_multiple_dispatch_view_native.h>
#include <ATen/ops/_test_optional_filled_intlist_native.h>
#include <ATen/ops/_test_optional_floatlist_native.h>
#include <ATen/ops/_test_optional_intlist_native.h>
#include <ATen/ops/_test_string_default_native.h>
#include <ATen/ops/_test_warn_in_autograd_native.h>
#include <ATen/ops/empty_like.h>
#endif

#include <c10/util/irange.h>

namespace at {
namespace native {

/// If addends is nullopt, return values.
/// Else, return a new tensor containing the elementwise sums.
Tensor _test_optional_intlist(
    const Tensor& values,
    at::OptionalIntArrayRef addends) {
  if (!addends) {
    return values;
  }
  TORCH_CHECK(values.dim() == 1);
  Tensor output = at::empty_like(values);
  auto inp = values.accessor<int,1>();
  auto out = output.accessor<int,1>();
  for (const auto i : c10::irange(values.size(0))) {
    out[i] = inp[i] + addends->at(i);
  }
  return output;
}

/// If addends is nullopt, return values.
/// Else, return a new tensor containing the elementwise sums.
Tensor _test_optional_floatlist(
    const Tensor& values,
    c10::optional<ArrayRef<double>> addends) {
  if (!addends) {
    return values;
  }
  TORCH_CHECK(values.dim() == 1);
  Tensor output = at::empty_like(values);
  auto inp = values.accessor<float,1>();
  auto out = output.accessor<float,1>();
  for (const auto i : c10::irange(values.size(0))) {
    out[i] = inp[i] + addends->at(i);
  }
  return output;
}

// Test default strings can handle escape sequences properly (although commas are broken)
Tensor _test_string_default(const Tensor& dummy, c10::string_view a, c10::string_view b) {
  const c10::string_view expect = "\"'\\";
  TORCH_CHECK(a == expect, "Default A failed");
  TORCH_CHECK(b == expect, "Default B failed");
  return dummy;
}

// Test that overloads with ambiguity created by defaulted parameters work.
// The operator declared first should have priority always

// Overload a
Tensor _test_ambiguous_defaults(const Tensor& dummy, int64_t a, int64_t b) {
  TORCH_CHECK(a == 1);
  TORCH_CHECK(b == 1);
  return c10::scalar_to_tensor(1);
}

// Overload b
Tensor _test_ambiguous_defaults(const Tensor& dummy, int64_t a, c10::string_view b) {
  TORCH_CHECK(a == 2);
  TORCH_CHECK(b == "2");
  return c10::scalar_to_tensor(2);
}

Tensor _test_warn_in_autograd(const Tensor &self) {
  return self.clone();
}

// Test registration of per-dispatch-key derivatives in derivatives.yaml.
// See derivatives.yaml for dummy registrations.

Tensor _test_autograd_multiple_dispatch_fullcoverage(const Tensor &self) {
  return self.clone();
}

Tensor _test_autograd_multiple_dispatch_ntonly(const Tensor &self, bool b) {
  return self.clone();
}

// Test derivative dispatch registration for view_copy ops
Tensor _test_autograd_multiple_dispatch_view(const Tensor &self) {
  return self.view(-1);
}

} // namespace native

namespace functionalization {

// view_copy ops must have a functional inverse registered
Tensor FunctionalInverses::_test_autograd_multiple_dispatch_view_copy_inverse(const at::Tensor& base, const at::Tensor& mutated_view, bool reapply_views) {
    TORCH_INTERNAL_ASSERT(false,
    "Attempted to call _test_autograd_multiple_dispatch_view_copy_inverse() during the functionalization pass. ",
    "This function is for testing only and should never be called.");
    return Tensor();
}

} // namespace functionalization
} // namespace at

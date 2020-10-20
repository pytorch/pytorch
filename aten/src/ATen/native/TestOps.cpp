// Copyright 2004-present Facebook. All Rights Reserved.

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

/// If addends is nullopt, return values.
/// Else, return a new tensor containing the elementwise sums.
Tensor _test_optional_intlist(
    const Tensor& values,
    c10::optional<IntArrayRef> addends) {
  if (!addends) {
    return values;
  }
  TORCH_CHECK(values.dim() == 1);
  Tensor output = at::empty_like(values);
  auto inp = values.accessor<int,1>();
  auto out = output.accessor<int,1>();
  for(int i = 0; i < values.size(0); ++i) {
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
  for(int i = 0; i < values.size(0); ++i) {
    out[i] = inp[i] + addends->at(i);
  }
  return output;
}

// Test default strings can handle escape sequences properly (although commas are broken)
Tensor _test_string_default(const Tensor& dummy, std::string a, std::string b) {
  const c10::string_view expect = "\"'\\";
  TORCH_CHECK(a == expect, "Default A failed");
  TORCH_CHECK(b == expect, "Default B failed");
  return dummy;
}

} // namespace native
} // namespace at

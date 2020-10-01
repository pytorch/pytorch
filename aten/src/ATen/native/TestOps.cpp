// Copyright 2004-present Facebook. All Rights Reserved.

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ScalarOps.h>

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

// Test that overloads with ambiguity created by defaulted parameters work.
// The operator declared first should have priority always

// Overload a
Tensor _test_ambiguous_defaults(const Tensor& dummy, int64_t a, std::string b, std::string c) {
  TORCH_CHECK(a == 1 || a == -1);
  TORCH_CHECK(b == "1" || b == "a");
  TORCH_CHECK(c == "1" || c == "a");
  return c10::scalar_to_tensor(1);
}

// Overload b
Tensor _test_ambiguous_defaults(const Tensor& dummy, int64_t a, std::string b, int64_t c) {
  TORCH_CHECK(a == 2 || a == -2);
  TORCH_CHECK(b == "2" || b == "b");
  TORCH_CHECK(c == 2 || c == -2);
  return c10::scalar_to_tensor(2);
}

// Overload c
Tensor _test_ambiguous_defaults(const Tensor& dummy, std::string a, std::string b) {
  TORCH_CHECK(a == "3" || a == "c");
  TORCH_CHECK(b == "3" || b == "c");
  return c10::scalar_to_tensor(3);
}

// Overload d
Tensor _test_ambiguous_defaults(const Tensor& dummy, int64_t a, std::string b, std::string c, std::string d) {
  TORCH_CHECK(a == 4 || a == -4);
  TORCH_CHECK(b == "4" || b == "d");
  TORCH_CHECK(c == "4" || c == "d");
  TORCH_CHECK(d == "4" || d == "d");
  return c10::scalar_to_tensor(4);
}

} // namespace native
} // namespace at

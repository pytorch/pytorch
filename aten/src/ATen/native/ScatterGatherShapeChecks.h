#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

namespace {

// Used for `gather`-like methods
// Test:
// 1. index.size(d) == self.size(d) for all d != dim
void gather_shape_check(const Tensor& self, int64_t dim, const Tensor& index);

// Used for `scatter` and `scatter_add`
// Tests:
//  1. index.size(d) <= self.size(d) for all d != dim
//  2. index.size(d) <= src.size(d) for all d if src is a Tensor
void scatter_shape_check(
  const Tensor& self, int64_t dim, const Tensor& index,
  const c10::optional<Tensor>& src_opt = c10::nullopt
);

} // anonymous namespace

}} // namespace at::native

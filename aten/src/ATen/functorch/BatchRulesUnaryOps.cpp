// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace at::functorch {

namespace{
std::tuple<Tensor,optional<int64_t>>
clone_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    optional<MemoryFormat> memory_format) {
  // Memory format support is a little tricky because vmap is allowed to move
  // around batch dimensions and some memory formats are rank-dependent.
  // Another weird case is:
  // - a tensor with MemoryFormat::ChannelsLast MUST have 4 dimensions. Do we
  //   allow the user to clone a Tensor with 3 logical dimensions and 1 batch
  //   dim into a ChannelsLast Tensor? What about a Tensor with 3 logical dims
  //   and N>1 batch dims?
  TORCH_CHECK(!memory_format.has_value() || memory_format == MemoryFormat::Preserve
      || memory_format == MemoryFormat::Contiguous,
      "NYI: Tensor.clone(memory_format) inside vmap is only supported with ",
      "memory_format torch.preserve_format or torch.contiguous_format (got ",
      *memory_format, ")");

  if (memory_format == MemoryFormat::Contiguous) {
    // There is an ambiguity here when the batch dims are not at the front of
    // the tensor.
    // >>> x = torch.randn(3, B0, 5)
    // >>> y = vmap(lambda x: x.clone(torch.contiguous_format), in_dims=1, out_dims=0)(x)
    // >>> y[0].is_contiguous()
    // ???
    // Should we make the whole tensor contiguous, or should we
    // make the non-batch dims contiguous? We've chosen the latter because
    // philosophically vmap hides the batch dims and operates on a per-sample level.
    auto self_ = moveBatchDimToFront(self, self_bdim);
    auto result = at::clone(self_, memory_format);
    return std::make_tuple(result, 0);
  }

  TORCH_INTERNAL_ASSERT(!memory_format.has_value() || memory_format == MemoryFormat::Preserve);
  auto result = at::clone(self, memory_format);
  return std::make_tuple(result, self_bdim);
}

std::tuple<Tensor,optional<int64_t>>
view_as_complex_batch_rule(const Tensor& self, optional<int64_t> self_bdim) {
  // guard against the user passing in a batch of scalar tensors with batch
  // size equal to 2.
  TORCH_CHECK(self.sizes().size() > 1, "Input tensor must have one or more dimensions");

  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto result = at::view_as_complex(self_);
  return std::make_tuple(result, 0);
}

std::tuple<Tensor,optional<int64_t>>
to_other_batch_rule(const Tensor& self, optional<int64_t> self_bdim,
                    const Tensor& other, optional<int64_t> other_bdim,
                    bool non_blocking,
                    bool copy, c10::optional<at::MemoryFormat> memory_format) {
  return std::make_tuple(self.to(other, non_blocking, copy, memory_format), self_bdim);
}
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {

#define UNARY_POINTWISE_ALL2(op, overload) \
  POINTWISE_BOXED2(op ## _, overload); \
  VMAP_SUPPORT2(op, overload, BASIC_UNARY_BATCH_RULE(ATEN_FN2(op, overload)));
#define UNARY_POINTWISE_ALL(op) \
  POINTWISE_BOXED(op ## _); \
  VMAP_SUPPORT(op, BASIC_UNARY_BATCH_RULE(ATEN_FN(op)));

  UNARY_POINTWISE(view_as_real);
  VMAP_SUPPORT(view_as_complex, view_as_complex_batch_rule);
  VMAP_SUPPORT(clone, clone_batch_rule);

  UNARY_POINTWISE(_to_copy);
  UNARY_POINTWISE(alias);
  UNARY_POINTWISE_ALL(abs);
  UNARY_POINTWISE_ALL(acos);
  UNARY_POINTWISE_ALL(acosh);
  UNARY_POINTWISE(angle);
  UNARY_POINTWISE_ALL(asin);
  UNARY_POINTWISE_ALL(asinh);
  UNARY_POINTWISE_ALL(atan);
  UNARY_POINTWISE_ALL(atanh);
  UNARY_POINTWISE_ALL(bitwise_not);
  UNARY_POINTWISE_ALL(ceil);
  UNARY_POINTWISE_ALL(cos);
  UNARY_POINTWISE_ALL(cosh);
  UNARY_POINTWISE(_conj);
  UNARY_POINTWISE_ALL(deg2rad);
  UNARY_POINTWISE(detach);
  UNARY_POINTWISE_ALL(digamma);
  UNARY_POINTWISE_ALL(erf);
  UNARY_POINTWISE_ALL(exp);
  UNARY_POINTWISE_ALL(expm1);
  UNARY_POINTWISE_ALL(floor);
  UNARY_POINTWISE_ALL(frac);
  UNARY_POINTWISE(isnan);
  UNARY_POINTWISE(isinf);
  UNARY_POINTWISE(isposinf);
  UNARY_POINTWISE(isneginf);
  UNARY_POINTWISE_ALL(lgamma);
  UNARY_POINTWISE_ALL(log);
  UNARY_POINTWISE_ALL(log10);
  UNARY_POINTWISE_ALL(log1p);
  UNARY_POINTWISE_ALL(log2);
  UNARY_POINTWISE_ALL(logical_not);
  UNARY_POINTWISE_ALL(logit);
  UNARY_POINTWISE_ALL(mish);
  UNARY_POINTWISE_ALL(mvlgamma);
  UNARY_POINTWISE_ALL(nan_to_num);
  UNARY_POINTWISE_ALL(neg);
  UNARY_POINTWISE_ALL(rad2deg);
  UNARY_POINTWISE_ALL(reciprocal);
  UNARY_POINTWISE_ALL(round);
  UNARY_POINTWISE_ALL2(round, decimals);
  UNARY_POINTWISE_ALL(rsqrt);
  UNARY_POINTWISE_ALL(sgn);
  UNARY_POINTWISE_ALL(sign);
  UNARY_POINTWISE(signbit);
  UNARY_POINTWISE_ALL(sin);
  UNARY_POINTWISE_ALL(sinc);
  UNARY_POINTWISE_ALL(sinh);
  UNARY_POINTWISE_ALL(sqrt);
  UNARY_POINTWISE_ALL(tan);
  UNARY_POINTWISE_ALL(threshold);
  UNARY_POINTWISE_ALL(trunc);

  // special-related
  UNARY_POINTWISE_ALL(i0);
  UNARY_POINTWISE_ALL(erfc);
  UNARY_POINTWISE_ALL(erfinv);
  UNARY_POINTWISE_ALL(exp2);

  // torch.special.* functions
  UNARY_POINTWISE(special_entr);
  UNARY_POINTWISE(special_erfcx);
  UNARY_POINTWISE(special_i0e);
  UNARY_POINTWISE(special_i1);
  UNARY_POINTWISE(special_i1e);
  UNARY_POINTWISE(special_ndtri);
  POINTWISE_BOXED(special_bessel_j0);
  POINTWISE_BOXED(special_spherical_bessel_j0);
  POINTWISE_BOXED(special_bessel_j1);
  POINTWISE_BOXED(special_modified_bessel_i0);
  POINTWISE_BOXED(special_modified_bessel_i1);
  POINTWISE_BOXED(special_scaled_modified_bessel_k0);
  POINTWISE_BOXED(special_modified_bessel_k0);
  POINTWISE_BOXED(special_scaled_modified_bessel_k1);
  POINTWISE_BOXED(special_modified_bessel_k1);
  POINTWISE_BOXED(special_bessel_y0);
  POINTWISE_BOXED(special_bessel_y1);

  // Activation functions (from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
  UNARY_POINTWISE_ALL(elu);
  UNARY_POINTWISE(hardshrink);
  UNARY_POINTWISE_ALL(hardsigmoid);
  UNARY_POINTWISE_ALL(hardtanh);
  UNARY_POINTWISE_ALL(hardswish);
  UNARY_POINTWISE_ALL(leaky_relu);
  UNARY_POINTWISE_ALL(relu);
  UNARY_POINTWISE_ALL(celu);
  UNARY_POINTWISE(gelu);
  UNARY_POINTWISE_ALL(sigmoid);
  UNARY_POINTWISE_ALL(silu);
  UNARY_POINTWISE(softplus);
  UNARY_POINTWISE(softshrink);
  UNARY_POINTWISE_ALL(tanh);

  POINTWISE_BOXED(fill_.Scalar);
  POINTWISE_BOXED(zero_);

#undef UNARY_POINTWISE
#undef UNARY_POINTWISE_ALL

}

#undef INVOKE
} // namespace at::functorch

// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_mm_reduce_impl_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/asin.h>
#include <ATen/ops/asin_native.h>
#include <ATen/ops/asinh.h>
#include <ATen/ops/asinh_native.h>
#include <ATen/ops/atan.h>
#include <ATen/ops/atan_native.h>
#include <ATen/ops/atanh.h>
#include <ATen/ops/atanh_native.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/ceil_native.h>
#include <ATen/ops/deg2rad.h>
#include <ATen/ops/deg2rad_native.h>
#include <ATen/ops/erf.h>
#include <ATen/ops/erf_native.h>
#include <ATen/ops/erfinv.h>
#include <ATen/ops/erfinv_native.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/expm1_native.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/floor_native.h>
#include <ATen/ops/frac.h>
#include <ATen/ops/frac_native.h>
#include <ATen/ops/isinf.h>
#include <ATen/ops/isinf_native.h>
#include <ATen/ops/isnan.h>
#include <ATen/ops/isnan_native.h>
#include <ATen/ops/isneginf.h>
#include <ATen/ops/isneginf_native.h>
#include <ATen/ops/isposinf.h>
#include <ATen/ops/isposinf_native.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/log1p_native.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/nan_to_num_native.h>
#include <ATen/ops/rad2deg.h>
#include <ATen/ops/rad2deg_native.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/round.h>
#include <ATen/ops/round_native.h>
#include <ATen/ops/sgn.h>
#include <ATen/ops/sgn_native.h>
#include <ATen/ops/sign.h>
#include <ATen/ops/sign_native.h>
#include <ATen/ops/signbit.h>
#include <ATen/ops/signbit_native.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sin_native.h>
#include <ATen/ops/sinh.h>
#include <ATen/ops/sinh_native.h>
#include <ATen/ops/sparse_resize_native.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/tan.h>
#include <ATen/ops/tan_native.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/tanh_native.h>
#include <ATen/ops/threshold_backward.h>
#include <ATen/ops/threshold_backward_native.h>
#include <ATen/ops/trunc.h>
#include <ATen/ops/trunc_native.h>
#include <ATen/ops/is_pinned_native.h>
#include <ATen/ops/_pin_memory_native.h>
#endif

namespace at::native {
namespace {

template <typename Ufunc>
Tensor coalesced_unary_ufunc(const Tensor &self, const Ufunc &ufunc) {
  TORCH_INTERNAL_ASSERT(self.is_sparse());
  const auto input = self.coalesce();
  Tensor out_values = ufunc(input.values());
  Tensor result = at::_sparse_coo_tensor_with_dims_and_tensors(
      input.sparse_dim(),
      input.dense_dim(),
      input.sizes(),
      input.indices().clone(),
      out_values,
      input.options().dtype(out_values.scalar_type()),
      /*is_coalesced=*/ true);
  return result;
}

template <typename Ufunc>
Tensor& coalesced_unary_ufunc_(Tensor &self, const Ufunc &ufunc) {
  TORCH_INTERNAL_ASSERT(self.is_sparse());
  auto values = self._values();
  ufunc(values);
  return self;
}

template <typename Ufunc>
Tensor& coalesced_unary_ufunc_out(const Tensor &self, Tensor &result, const Ufunc &ufunc) {
  if (self.is_same(result)) {
    TORCH_CHECK(self.is_coalesced(), "expected coalesced tensor for inplace operation");
    auto values = self._values();
    ufunc(values, values);
    return result;
  }

  TORCH_CHECK(self.is_sparse() && result.is_sparse());
  const auto input = self.coalesce();
  sparse_resize_(result, input.sizes(), input.sparse_dim(), input.dense_dim());
  auto *input_impl = sparse::get_sparse_impl(input);
  auto *result_impl = sparse::get_sparse_impl(result);

  auto input_values = input_impl->values();
  auto result_values = result_impl->values();
  result_values.resize_(input_values.sizes());
  ufunc(input_values, result_values);

  auto input_indices = input_impl->indices();
  auto result_indices = result_impl->indices();
  result_indices.resize_(input_indices.sizes());
  result_indices.copy_(input_indices);
  result._coalesced_(true);
  return result;
}

}  // namespace (anonymous)

// Generic formulation for unary operators which map 0 -> 0 so
// we can just transform self.values() and preserve the sparsity pattern.
//
// Any non-linear function requires the tensor to be coalesced before
// we can calculate the result. This also means inplace calculations
// are only possible on coalesced tensors.

#define COALESCED_UNARY_UFUNC_FUNCTIONAL(op_name)   \
  Tensor op_name##_sparse(const Tensor &self) {     \
    return coalesced_unary_ufunc(                   \
        self, [](const Tensor &t) {                 \
          return at::op_name(t);                    \
        });                                         \
  }

#define COALESCED_UNARY_UFUNC_NO_INPLACE(op_name)                       \
  COALESCED_UNARY_UFUNC_FUNCTIONAL(op_name)                             \
  Tensor& op_name##_sparse_out(const Tensor &self,                      \
                               Tensor &out) {                           \
    return coalesced_unary_ufunc_out(                                   \
        self, out, [](const Tensor &t, Tensor &out) {                   \
          return at::op_name##_outf(t, out);                            \
        });                                                             \
  }

#define COALESCED_UNARY_UFUNC(op_name)                                  \
  COALESCED_UNARY_UFUNC_NO_INPLACE(op_name)                             \
  Tensor& op_name##_sparse_(Tensor &self) {                             \
    TORCH_CHECK(self.is_coalesced(),                                    \
                #op_name "_ requires coalesced input");                 \
    return coalesced_unary_ufunc_(self, [](Tensor &t) {                 \
      return t.op_name##_();                                            \
    });                                                                 \
  }

COALESCED_UNARY_UFUNC(abs)
COALESCED_UNARY_UFUNC(asin)
COALESCED_UNARY_UFUNC(asinh)
COALESCED_UNARY_UFUNC(atan)
COALESCED_UNARY_UFUNC(atanh)
COALESCED_UNARY_UFUNC(ceil)
COALESCED_UNARY_UFUNC(deg2rad)
COALESCED_UNARY_UFUNC(erf)
COALESCED_UNARY_UFUNC(erfinv)
COALESCED_UNARY_UFUNC(expm1)
COALESCED_UNARY_UFUNC(floor)
COALESCED_UNARY_UFUNC(frac)
COALESCED_UNARY_UFUNC(log1p)
COALESCED_UNARY_UFUNC(round)
COALESCED_UNARY_UFUNC(rad2deg)
COALESCED_UNARY_UFUNC(sign)
COALESCED_UNARY_UFUNC(sgn)
COALESCED_UNARY_UFUNC(sin)
COALESCED_UNARY_UFUNC(sinh)
COALESCED_UNARY_UFUNC(sqrt)
COALESCED_UNARY_UFUNC(tan)
COALESCED_UNARY_UFUNC(tanh)
COALESCED_UNARY_UFUNC(trunc)
// relu function has no declaration, it may be unused in Pytorch.
// But we keep it and ignore the warning here until verified in the future.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"
COALESCED_UNARY_UFUNC(relu)
#pragma clang diagnostic pop

COALESCED_UNARY_UFUNC_NO_INPLACE(signbit)
COALESCED_UNARY_UFUNC_NO_INPLACE(isneginf)
COALESCED_UNARY_UFUNC_NO_INPLACE(isposinf)

COALESCED_UNARY_UFUNC_FUNCTIONAL(isnan)
COALESCED_UNARY_UFUNC_FUNCTIONAL(isinf)

Tensor isinf_sparse_meta(const Tensor& self) {
  TORCH_CHECK_NOT_IMPLEMENTED(0, "nyi isinf for SparseMeta");
}

// Threshold_backward is not unary but it is the backward used for relu which is
// unary
Tensor threshold_backward_sparse(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold) {
  const auto grad = [&]() {
    if (!grad_output._nnz() && self._nnz() > 0) {
      return at::sparse::zeros_like_with_indices(self);
    } else {
      return grad_output;
    }
  }();
  const auto self_v = [&self]() {
    if (self.is_coalesced()) {
      return self.values();
    } else {
      return self.coalesce().values();
    }
  }();
  return coalesced_unary_ufunc(grad, [&](const Tensor& t) {
    return at::threshold_backward(t, self_v, threshold);
  });
}

Tensor& threshold_backward_sparse_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& grad_input) {
  const auto grad = [&]() {
    if (!grad_output._nnz() && self._nnz() > 0) {
      return at::sparse::zeros_like_with_indices(self);
    } else {
      return grad_output;
    }
  }();
  auto self_v = [&self]() {
    if (self.is_coalesced()) {
      return self.values();
    } else {
      return self.coalesce().values();
    }
  }();
  return coalesced_unary_ufunc_out(
      grad, grad_input, [&](const Tensor& t, Tensor& out) {
        return at::threshold_backward_outf(t, self_v, threshold, out);
      });
}

Tensor nan_to_num_sparse(
    const Tensor &self, std::optional<double> nan,
    std::optional<double> posinf, std::optional<double> neginf) {
  return coalesced_unary_ufunc(
      self, [&](const Tensor &t) {
        return at::nan_to_num(t, nan, posinf, neginf);
      });
}
Tensor& nan_to_num_sparse_out(
    const Tensor &self, std::optional<double> nan,
    std::optional<double> posinf, std::optional<double> neginf,
    Tensor &out) {
  return coalesced_unary_ufunc_out(
      self, out, [&](const Tensor &t, Tensor &out) {
        return at::nan_to_num_outf(t, nan, posinf, neginf, out);
      });
}
Tensor& nan_to_num_sparse_(
    Tensor &self, std::optional<double> nan,
    std::optional<double> posinf, std::optional<double> neginf) {
  TORCH_CHECK(self.is_coalesced(), "nan_to_num_ requires coalesced input");
  return nan_to_num_sparse_out(self, nan, posinf, neginf, self);
}

}  // namespace at::native

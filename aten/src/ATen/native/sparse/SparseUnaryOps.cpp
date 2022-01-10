#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>

namespace at {
namespace native {
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
      input.options().dtype(out_values.scalar_type()));
  result._coalesced_(true);
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

COALESCED_UNARY_UFUNC(abs);
COALESCED_UNARY_UFUNC(asin);
COALESCED_UNARY_UFUNC(asinh);
COALESCED_UNARY_UFUNC(atan);
COALESCED_UNARY_UFUNC(atanh);
COALESCED_UNARY_UFUNC(ceil);
COALESCED_UNARY_UFUNC(erf);
COALESCED_UNARY_UFUNC(erfinv);
COALESCED_UNARY_UFUNC(expm1);
COALESCED_UNARY_UFUNC(floor);
COALESCED_UNARY_UFUNC(log1p);
COALESCED_UNARY_UFUNC(round);
COALESCED_UNARY_UFUNC(sign);
COALESCED_UNARY_UFUNC(sgn);
COALESCED_UNARY_UFUNC(sin);
COALESCED_UNARY_UFUNC(sinh);
COALESCED_UNARY_UFUNC(sqrt);
COALESCED_UNARY_UFUNC(tan);
COALESCED_UNARY_UFUNC(tanh);
COALESCED_UNARY_UFUNC(trunc);

COALESCED_UNARY_UFUNC_NO_INPLACE(signbit);
COALESCED_UNARY_UFUNC_NO_INPLACE(isneginf);
COALESCED_UNARY_UFUNC_NO_INPLACE(isposinf);

COALESCED_UNARY_UFUNC_FUNCTIONAL(isnan);
COALESCED_UNARY_UFUNC_FUNCTIONAL(isinf);

Tensor nan_to_num_sparse(
    const Tensor &self, c10::optional<double> nan,
    c10::optional<double> posinf, c10::optional<double> neginf) {
  return coalesced_unary_ufunc(
      self, [&](const Tensor &t) {
        return at::nan_to_num(t, nan, posinf, neginf);
      });
}
Tensor& nan_to_num_sparse_out(
    const Tensor &self, c10::optional<double> nan,
    c10::optional<double> posinf, c10::optional<double> neginf,
    Tensor &out) {
  return coalesced_unary_ufunc_out(
      self, out, [&](const Tensor &t, Tensor &out) {
        return at::nan_to_num_outf(t, nan, posinf, neginf, out);
      });
}
Tensor& nan_to_num_sparse_(
    Tensor &self, c10::optional<double> nan,
    c10::optional<double> posinf, c10::optional<double> neginf) {
  TORCH_CHECK(self.is_coalesced(), "nan_to_num_ requires coalesced input");
  return nan_to_num_sparse_out(self, nan, posinf, neginf, self);
}

}}  // namespace at::native

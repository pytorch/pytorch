#include <vector>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/ForeachUtils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#else
#include <ATen/ops/_foreach_abs_native.h>
#include <ATen/ops/_foreach_acos_native.h>
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_addcdiv_native.h>
#include <ATen/ops/_foreach_addcmul_native.h>
#include <ATen/ops/_foreach_asin_native.h>
#include <ATen/ops/_foreach_atan_native.h>
#include <ATen/ops/_foreach_ceil_native.h>
#include <ATen/ops/_foreach_clamp_max_native.h>
#include <ATen/ops/_foreach_clamp_min_native.h>
#include <ATen/ops/_foreach_copy_native.h>
#include <ATen/ops/_foreach_cos_native.h>
#include <ATen/ops/_foreach_cosh_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_erf_native.h>
#include <ATen/ops/_foreach_erfc_native.h>
#include <ATen/ops/_foreach_exp_native.h>
#include <ATen/ops/_foreach_expm1_native.h>
#include <ATen/ops/_foreach_floor_native.h>
#include <ATen/ops/_foreach_frac_native.h>
#include <ATen/ops/_foreach_lerp_native.h>
#include <ATen/ops/_foreach_lgamma_native.h>
#include <ATen/ops/_foreach_log10_native.h>
#include <ATen/ops/_foreach_log1p_native.h>
#include <ATen/ops/_foreach_log2_native.h>
#include <ATen/ops/_foreach_log_native.h>
#include <ATen/ops/_foreach_max_native.h>
#include <ATen/ops/_foreach_maximum_native.h>
#include <ATen/ops/_foreach_minimum_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_neg_native.h>
#include <ATen/ops/_foreach_norm_native.h>
#include <ATen/ops/_foreach_pow_native.h>
#include <ATen/ops/_foreach_reciprocal_native.h>
#include <ATen/ops/_foreach_round_native.h>
#include <ATen/ops/_foreach_rsqrt_native.h>
#include <ATen/ops/_foreach_sigmoid_native.h>
#include <ATen/ops/_foreach_sign_native.h>
#include <ATen/ops/_foreach_sin_native.h>
#include <ATen/ops/_foreach_sinh_native.h>
#include <ATen/ops/_foreach_sqrt_native.h>
#include <ATen/ops/_foreach_sub_native.h>
#include <ATen/ops/_foreach_tan_native.h>
#include <ATen/ops/_foreach_tanh_native.h>
#include <ATen/ops/_foreach_trunc_native.h>
#include <ATen/ops/_foreach_where_native.h>
#include <ATen/ops/_foreach_zero_native.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/max.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/where.h>
#endif

namespace at::native {

#define FOREACH_BINARY_OP_TENSOR(OP)                            \
  void foreach_tensor_##OP##_tensor_kernel_slow_(               \
      TensorList tensors, const Tensor& scalar) {               \
    TORCH_CHECK(                                                \
        scalar.dim() == 0 && scalar.numel() == 1,               \
        "scalar tensor expected to be 0 dim but it has ",       \
        scalar.dim(),                                           \
        " dimensions and ",                                     \
        scalar.numel(),                                         \
        " elements.");                                          \
    check_foreach_api_restrictions(tensors);                    \
                                                                \
    for (auto& t : tensors) {                                   \
      t.OP##_(scalar);                                          \
    }                                                           \
  }                                                             \
                                                                \
  std::vector<Tensor> foreach_tensor_##OP##_tensor_kernel_slow( \
      TensorList tensors, const Tensor& scalar) {               \
    TORCH_CHECK(                                                \
        scalar.dim() == 0 && scalar.numel() == 1,               \
        "scalar tensor expected to be 0 dim but it has ",       \
        scalar.dim(),                                           \
        " dimensions and ",                                     \
        scalar.numel(),                                         \
        " elements.");                                          \
    check_foreach_api_restrictions(tensors);                    \
                                                                \
    std::vector<Tensor> result;                                 \
    result.reserve(tensors.size());                             \
    for (const auto& t : tensors) {                             \
      result.emplace_back(t.OP(scalar));                        \
    }                                                           \
                                                                \
    return result;                                              \
  }

#define FOREACH_BINARY_OP_TENSOR_ALPHA(OP)                             \
  void foreach_tensor_##OP##_tensor_kernel_slow_(                      \
      TensorList tensors, const Tensor& scalar, const Scalar& alpha) { \
    TORCH_CHECK(                                                       \
        scalar.dim() == 0 && scalar.numel() == 1,                      \
        "scalar tensor expected to be 0 dim but it has ",              \
        scalar.dim(),                                                  \
        " dimensions and ",                                            \
        scalar.numel(),                                                \
        " elements.");                                                 \
    check_foreach_api_restrictions(tensors);                           \
                                                                       \
    for (auto& t : tensors) {                                          \
      t.OP##_(scalar, alpha);                                          \
    }                                                                  \
  }                                                                    \
                                                                       \
  std::vector<Tensor> foreach_tensor_##OP##_tensor_kernel_slow(        \
      TensorList tensors, const Tensor& scalar, const Scalar& alpha) { \
    TORCH_CHECK(                                                       \
        scalar.dim() == 0 && scalar.numel() == 1,                      \
        "scalar tensor expected to be 0 dim but it has ",              \
        scalar.dim(),                                                  \
        " dimensions and ",                                            \
        scalar.numel(),                                                \
        " elements.");                                                 \
    check_foreach_api_restrictions(tensors);                           \
                                                                       \
    std::vector<Tensor> result;                                        \
    result.reserve(tensors.size());                                    \
    for (const auto& t : tensors) {                                    \
      result.emplace_back(t.OP(scalar, alpha));                        \
    }                                                                  \
                                                                       \
    return result;                                                     \
  }

#define FOREACH_BINARY_OP_SCALAR(OP)                            \
  void foreach_tensor_##OP##_scalar_kernel_slow_(               \
      TensorList tensors, const Scalar& scalar) {               \
    check_foreach_api_restrictions(tensors);                    \
                                                                \
    for (auto& t : tensors) {                                   \
      t.OP##_(scalar);                                          \
    }                                                           \
  }                                                             \
                                                                \
  std::vector<Tensor> foreach_tensor_##OP##_scalar_kernel_slow( \
      TensorList tensors, const Scalar& scalar) {               \
    check_foreach_api_restrictions(tensors);                    \
                                                                \
    std::vector<Tensor> result;                                 \
    result.reserve(tensors.size());                             \
    for (const auto& t : tensors) {                             \
      result.emplace_back(t.OP(scalar));                        \
    }                                                           \
                                                                \
    return result;                                              \
  }

#define FOREACH_BINARY_OP_SCALARLIST(OP)                            \
  void foreach_tensor_##OP##_scalarlist_kernel_slow_(               \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {           \
    check_foreach_api_restrictions(tensors, scalars);               \
                                                                    \
    for (const auto i : c10::irange(tensors.size())) {              \
      tensors[i].OP##_(scalars[i]);                                 \
    }                                                               \
  }                                                                 \
                                                                    \
  std::vector<Tensor> foreach_tensor_##OP##_scalarlist_kernel_slow( \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {           \
    check_foreach_api_restrictions(tensors, scalars);               \
    std::vector<Tensor> result;                                     \
    result.reserve(tensors.size());                                 \
    for (const auto i : c10::irange(tensors.size())) {              \
      result.emplace_back(tensors[i].OP(scalars[i]));               \
    }                                                               \
                                                                    \
    return result;                                                  \
  }

#define FOREACH_BINARY_OP_LIST(OP)                            \
  std::vector<Tensor> foreach_tensor_##OP##_list_kernel_slow( \
      TensorList tensors1, TensorList tensors2) {             \
    check_foreach_api_restrictions(tensors1, tensors2);       \
                                                              \
    std::vector<Tensor> result;                               \
    result.reserve(tensors1.size());                          \
    for (const auto i : c10::irange(tensors1.size())) {       \
      result.emplace_back(tensors1[i].OP(tensors2[i]));       \
    }                                                         \
                                                              \
    return result;                                            \
  }                                                           \
                                                              \
  void foreach_tensor_##OP##_list_kernel_slow_(               \
      TensorList tensors1, TensorList tensors2) {             \
    check_foreach_api_restrictions(tensors1, tensors2);       \
                                                              \
    for (const auto i : c10::irange(tensors1.size())) {       \
      tensors1[i].OP##_(tensors2[i]);                         \
    }                                                         \
  }

#define FOREACH_BINARY_OP_LIST_ALPHA(OP)                               \
  std::vector<Tensor> foreach_tensor_##OP##_list_kernel_slow(          \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
                                                                       \
    std::vector<Tensor> result;                                        \
    result.reserve(tensors1.size());                                   \
    for (const auto i : c10::irange(tensors1.size())) {                \
      result.emplace_back(tensors1[i].OP(tensors2[i], alpha));         \
    }                                                                  \
                                                                       \
    return result;                                                     \
  }                                                                    \
                                                                       \
  void foreach_tensor_##OP##_list_kernel_slow_(                        \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
                                                                       \
    for (const auto i : c10::irange(tensors1.size())) {                \
      tensors1[i].OP##_(tensors2[i], alpha);                           \
    }                                                                  \
  }

#define FOREACH_UNARY_OP(OP)                                           \
  std::vector<Tensor> foreach_tensor_##OP##_slow(TensorList tensors) { \
    check_foreach_api_restrictions(tensors);                           \
                                                                       \
    std::vector<Tensor> result;                                        \
    result.reserve(tensors.size());                                    \
    for (const auto& t : tensors) {                                    \
      result.emplace_back(t.OP());                                     \
    }                                                                  \
                                                                       \
    return result;                                                     \
  }                                                                    \
                                                                       \
  void foreach_tensor_##OP##_slow_(TensorList tensors) {               \
    check_foreach_api_restrictions(tensors);                           \
                                                                       \
    for (auto& t : tensors) {                                          \
      t.OP##_();                                                       \
    }                                                                  \
  }

#define FOREACH_POINTWISE_OP_SCALAR(OP)                                   \
  std::vector<Tensor> foreach_tensor_##OP##_scalar_slow(                  \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Scalar& scalar) {                                             \
    check_foreach_api_restrictions(input, tensors1, tensors2);            \
                                                                          \
    std::vector<Tensor> result;                                           \
    for (const auto i : c10::irange(input.size())) {                      \
      result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalar)); \
    }                                                                     \
                                                                          \
    return result;                                                        \
  }                                                                       \
                                                                          \
  void foreach_tensor_##OP##_scalar_slow_(                                \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Scalar& scalar) {                                             \
    check_foreach_api_restrictions(input, tensors1, tensors2);            \
                                                                          \
    for (const auto i : c10::irange(input.size())) {                      \
      input[i].OP##_(tensors1[i], tensors2[i], scalar);                   \
    }                                                                     \
  }

#define FOREACH_POINTWISE_OP_SCALARLIST(OP)                                   \
  std::vector<Tensor> foreach_tensor_##OP##_scalarlist_slow(                  \
      TensorList input,                                                       \
      TensorList tensors1,                                                    \
      TensorList tensors2,                                                    \
      at::ArrayRef<Scalar> scalars) {                                         \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);       \
                                                                              \
    std::vector<Tensor> result;                                               \
    for (const auto i : c10::irange(input.size())) {                          \
      result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalars[i])); \
    }                                                                         \
                                                                              \
    return result;                                                            \
  }                                                                           \
                                                                              \
  void foreach_tensor_##OP##_scalarlist_slow_(                                \
      TensorList input,                                                       \
      TensorList tensors1,                                                    \
      TensorList tensors2,                                                    \
      at::ArrayRef<Scalar> scalars) {                                         \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);       \
                                                                              \
    for (const auto i : c10::irange(input.size())) {                          \
      input[i].OP##_(tensors1[i], tensors2[i], scalars[i]);                   \
    }                                                                         \
  }

#define FOREACH_POINTWISE_OP_TENSOR(OP)                                   \
  std::vector<Tensor> foreach_tensor_##OP##_tensor_slow(                  \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Tensor& scalars_) {                                           \
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size()); \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);   \
    return foreach_tensor_##OP##_scalarlist_slow(                         \
        input, tensors1, tensors2, scalars);                              \
  }                                                                       \
                                                                          \
  void foreach_tensor_##OP##_tensor_slow_(                                \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Tensor& scalars_) {                                           \
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size()); \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);   \
    foreach_tensor_##OP##_scalarlist_slow_(                               \
        input, tensors1, tensors2, scalars);                              \
  }

FOREACH_BINARY_OP_LIST_ALPHA(add)
FOREACH_BINARY_OP_LIST_ALPHA(sub)
FOREACH_BINARY_OP_LIST_ALPHA(lerp)

FOREACH_BINARY_OP_TENSOR_ALPHA(add)
FOREACH_BINARY_OP_TENSOR(mul)
FOREACH_BINARY_OP_TENSOR(div)

FOREACH_BINARY_OP_SCALAR(add)
FOREACH_BINARY_OP_SCALAR(sub)
FOREACH_BINARY_OP_SCALAR(mul)
FOREACH_BINARY_OP_SCALAR(div)
FOREACH_BINARY_OP_SCALAR(clamp_min)
FOREACH_BINARY_OP_SCALAR(clamp_max)
FOREACH_BINARY_OP_SCALAR(pow)

FOREACH_BINARY_OP_SCALARLIST(add)
FOREACH_BINARY_OP_SCALARLIST(sub)
FOREACH_BINARY_OP_SCALARLIST(mul)
FOREACH_BINARY_OP_SCALARLIST(div)
FOREACH_BINARY_OP_SCALARLIST(clamp_min)
FOREACH_BINARY_OP_SCALARLIST(clamp_max)
FOREACH_BINARY_OP_SCALARLIST(pow)

FOREACH_BINARY_OP_LIST(mul)
FOREACH_BINARY_OP_LIST(div)
FOREACH_BINARY_OP_LIST(clamp_min)
FOREACH_BINARY_OP_LIST(clamp_max)
FOREACH_BINARY_OP_LIST(pow)
// _foreach_copy_
void foreach_tensor_copy_list_kernel_slow_(
    TensorList self,
    TensorList src,
    const bool non_blocking) {
  check_foreach_api_restrictions(self, src);

  for (const auto i : c10::irange(self.size())) {
    self[i].copy_(src[i], non_blocking);
  }
}

FOREACH_UNARY_OP(sqrt)
FOREACH_UNARY_OP(exp)
FOREACH_UNARY_OP(abs)
FOREACH_UNARY_OP(acos)
FOREACH_UNARY_OP(asin)
FOREACH_UNARY_OP(atan)
FOREACH_UNARY_OP(ceil)
FOREACH_UNARY_OP(cos)
FOREACH_UNARY_OP(cosh)
FOREACH_UNARY_OP(erf)
FOREACH_UNARY_OP(erfc)
FOREACH_UNARY_OP(expm1)
FOREACH_UNARY_OP(floor)
FOREACH_UNARY_OP(log)
FOREACH_UNARY_OP(log10)
FOREACH_UNARY_OP(log1p)
FOREACH_UNARY_OP(log2)
FOREACH_UNARY_OP(neg)
FOREACH_UNARY_OP(tan)
FOREACH_UNARY_OP(tanh)
FOREACH_UNARY_OP(sin)
FOREACH_UNARY_OP(sinh)
FOREACH_UNARY_OP(round)
FOREACH_UNARY_OP(rsqrt)
FOREACH_UNARY_OP(lgamma)
FOREACH_UNARY_OP(frac)
FOREACH_UNARY_OP(trunc)
FOREACH_UNARY_OP(reciprocal)
FOREACH_UNARY_OP(sigmoid)
FOREACH_UNARY_OP(sign)

FOREACH_POINTWISE_OP_SCALAR(addcdiv)
FOREACH_POINTWISE_OP_SCALAR(addcmul)

FOREACH_POINTWISE_OP_SCALARLIST(addcdiv)
FOREACH_POINTWISE_OP_SCALARLIST(addcmul)

FOREACH_POINTWISE_OP_TENSOR(addcdiv)
FOREACH_POINTWISE_OP_TENSOR(addcmul)

std::vector<Tensor> foreach_tensor_ternary_lerp_slow(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  std::vector<Tensor> result;
  for (const auto i : c10::irange(tensors1.size())) {
    result.emplace_back(tensors1[i].lerp(tensors2[i], tensors3[i]));
  }
  return result;
}

void foreach_tensor_ternary_lerp_slow_(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  for (const auto i : c10::irange(tensors1.size())) {
    tensors1[i].lerp_(tensors2[i], tensors3[i]);
  }
}

std::vector<Tensor> foreach_tensor_lerp_scalarlist_kernel_slow(
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors1, tensors2, scalars);
  std::vector<Tensor> result;
  for (const auto i : c10::irange(tensors1.size())) {
    result.emplace_back(tensors1[i].lerp(tensors2[i], scalars[i]));
  }
  return result;
}

void foreach_tensor_lerp_scalarlist_kernel_slow_(
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors1, tensors2, scalars);
  for (const auto i : c10::irange(tensors1.size())) {
    tensors1[i].lerp_(tensors2[i], scalars[i]);
  }
}

void foreach_tensor_zero_slow_(TensorList tensors) {
  check_foreach_api_restrictions(tensors);

  for (auto& t : tensors) {
    t.zero_();
  }
}

std::vector<Tensor> foreach_tensor_norm_slow(
    TensorList tensors,
    const Scalar& ord,
    std::optional<ScalarType> dtype) {
  check_foreach_api_restrictions(tensors);
  std::vector<Tensor> result;
  for (const auto& t : tensors) {
    result.emplace_back(at::linalg_vector_norm(t, ord, {}, false, dtype));
  }
  return result;
}

std::vector<Tensor> foreach_tensor_max_slow(TensorList tensors) {
  check_foreach_api_restrictions(tensors);
  std::vector<Tensor> result;
  for (const auto& t : tensors) {
    result.emplace_back(at::max(t));
  }
  return result;
}

std::vector<Tensor> foreach_scalar_pow_list_kernel_slow(
    const Scalar& self,
    TensorList exponent) {
  check_foreach_api_restrictions(exponent);
  std::vector<Tensor> result;
  result.reserve(exponent.size());
  for (const auto& t : exponent) {
    result.emplace_back(at::pow(self, t));
  }
  return result;
}

std::vector<Tensor> foreach_tensor_where_scalar_slow(
    TensorList conditions,
    TensorList tensors,
    const at::Scalar& other) {
  check_foreach_api_restrictions(tensors);
  std::vector<Tensor> result;
  result.reserve(tensors.size());
  for (int64_t i = 0; i < tensors.size(); i++) {
    result.emplace_back(at::where(conditions[i], tensors[i], other));
  }
  return result;
}

} // namespace at::native

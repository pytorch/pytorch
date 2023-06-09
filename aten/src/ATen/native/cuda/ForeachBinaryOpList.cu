#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/ForeachMinMaxFunctors.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_clamp_max_native.h>
#include <ATen/ops/_foreach_clamp_min_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_pow_native.h>
#include <ATen/ops/_foreach_sub_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T, template <class> class Op>
std::vector<Tensor> foreach_tensor_list_op(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<3>(
      tensor_lists,
      BinaryOpListAlphaFunctor<
          T,
          /* depth */ 3,
          /* r_args_depth */ 2,
          /* res_arg_index */ 2>(),
      Op<opmath_t>(),
      alpha.to<opmath_t>());

  return tensor_lists[2];
}

template <typename T, template <class> class Op>
void foreach_tensor_list_op_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<2>(
      tensor_lists,
      BinaryOpListAlphaFunctor<
          T,
          /* depth */ 2,
          /* r_args_depth */ 2,
          /* res_arg_index */ 0>(),
      Op<opmath_t>(),
      alpha.to<opmath_t>());
  increment_version(tensors1);
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda",
      [&]() {
        return foreach_tensor_list_op<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda_",
      [&]() {
        foreach_tensor_list_op_<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
std::vector<Tensor> all_types_half_bfloat16(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda",
      [&]() {
        return foreach_tensor_list_op<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
void all_types_complex_half_bfloat16_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda_",
      [&]() {
        foreach_tensor_list_op_<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
void all_types_half_bfloat16_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda_",
      [&]() {
        foreach_tensor_list_op_<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_half_bfloat16(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda",
      [&]() {
        return foreach_tensor_list_op<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

#define FOREACH_BINARY_OP_LIST(FUNCTION, NAME, OP, DIVISION_OP)     \
  void foreach_tensor_##NAME##_list_kernel_cuda_(                   \
      TensorList tensors1, TensorList tensors2) {                   \
    check_foreach_api_restrictions(tensors1, tensors2);             \
    if (!can_use_fast_route(tensors1, tensors2, DIVISION_OP)) {     \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow_( \
          tensors1, tensors2);                                      \
    }                                                               \
                                                                    \
    FUNCTION##_<OP>(tensors1, tensors2);                            \
  }                                                                 \
                                                                    \
  std::vector<Tensor> foreach_tensor_##NAME##_list_kernel_cuda(     \
      TensorList tensors1, TensorList tensors2) {                   \
    check_foreach_api_restrictions(tensors1, tensors2);             \
    if (!can_use_fast_route(tensors1, tensors2, DIVISION_OP)) {     \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(  \
          tensors1, tensors2);                                      \
    }                                                               \
                                                                    \
    return FUNCTION<OP>(tensors1, tensors2);                        \
  }

#define FOREACH_BINARY_OP_LIST_ALPHA(FUNCTION, NAME, OP)               \
  void foreach_tensor_##NAME##_list_kernel_cuda_(                      \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
    if (!can_use_fast_route({tensors1, tensors2}, alpha)) {            \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow_(    \
          tensors1, tensors2, alpha);                                  \
    }                                                                  \
                                                                       \
    FUNCTION##_<OP>(tensors1, tensors2, alpha);                        \
  }                                                                    \
                                                                       \
  std::vector<Tensor> foreach_tensor_##NAME##_list_kernel_cuda(        \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
    if (!can_use_fast_route({tensors1, tensors2}, alpha)) {            \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(     \
          tensors1, tensors2, alpha);                                  \
    }                                                                  \
                                                                       \
    return FUNCTION<OP>(tensors1, tensors2, alpha);                    \
  }

FOREACH_BINARY_OP_LIST_ALPHA(
    all_types_complex_bool_half_bfloat16,
    add,
    std::plus);
FOREACH_BINARY_OP_LIST_ALPHA(
    all_types_complex_bool_half_bfloat16,
    sub,
    std::minus);
FOREACH_BINARY_OP_LIST(
    all_types_complex_bool_half_bfloat16,
    mul,
    std::multiplies,
    /*division_op*/ false);
FOREACH_BINARY_OP_LIST(
    all_types_complex_bool_half_bfloat16,
    div,
    std::divides,
    /*division_op*/ true);
FOREACH_BINARY_OP_LIST(
    all_types_half_bfloat16,
    clamp_max,
    minimum,
    /*division_op*/ false);
FOREACH_BINARY_OP_LIST(
    all_types_half_bfloat16,
    clamp_min,
    maximum,
    /*division_op*/ false);
// NOTE(crcrpar): [Why is foreach_pow's division_op=true?]
// To push integer inputs to slow path. This is because with integer type inputs
// the fast path behaves differently from the slow one. Need to investigate
// later.
FOREACH_BINARY_OP_LIST(
    all_types_complex_half_bfloat16,
    pow,
    power_functor,
    /*division_op*/ true);

} // namespace at::native

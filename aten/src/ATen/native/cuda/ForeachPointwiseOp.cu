#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_addcdiv_native.h>
#include <ATen/ops/_foreach_addcmul_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_maximum_native.h>
#include <ATen/ops/_foreach_minimum_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_sub_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

// Helper to check if all tensors in a list are 0D (scalar tensors)
inline bool all_tensors_are_0d(TensorList tensors) {
  return std::all_of(tensors.begin(), tensors.end(), [](const Tensor& t) {
    return t.dim() == 0;
  });
}

// Helper to check if all tensors have the same dtype as reference
inline bool all_same_dtype(TensorList tensors, ScalarType dtype) {
  return std::all_of(tensors.begin(), tensors.end(), [dtype](const Tensor& t) {
    return t.scalar_type() == dtype;
  });
}

inline bool all_same_device(TensorList tensors, Device device) {
  return std::all_of(tensors.begin(), tensors.end(), [device](const Tensor& t) {
    return t.device() == device;
  });
}

// Inplace variant for when tensor1 is a list of 0D tensors (scalars)
template <template <class> class Op>
void foreach_pointwise_op_0d_tensor1_(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha) {
  std::vector<std::vector<at::Tensor>> tensor_lists;

  // tensor_lists: input, tensor1 (0D), tensor2
  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_0d_tensor1__cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            PointwiseOpScalar0dTensorFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 2,
                /* res_arg_index */ 0>(),
            Op<opmath_t>(),
            alpha.to<opmath_t>());
      });
  increment_version(input);
}

template <template <class> class Op>
std::vector<Tensor> foreach_pointwise_op(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<4>(
            tensor_lists,
            PointwiseOpScalarFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            Op<opmath_t>(),
            scalar.to<opmath_t>());
      });

  return std::move(tensor_lists[3]);
}

// Variant for when tensor1 is a list of 0D tensors (scalars)
// tensor_lists: input, tensor1 (0D), tensor2, output
// The 0D tensor1 values are loaded from device memory in the functor
template <template <class> class Op>
std::vector<Tensor> foreach_pointwise_op_0d_tensor1(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  // tensor_lists: input, tensor1 (0D), tensor2, output
  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_0d_tensor1_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<4>(
            tensor_lists,
            PointwiseOpScalar0dTensorFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 2,
                /* res_arg_index */ 3>(),
            Op<opmath_t>(),
            alpha.to<opmath_t>());
      });

  return std::move(tensor_lists[3]);
}

template <template <class> class Op>
void foreach_pointwise_op_(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op__cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            PointwiseOpScalarFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            Op<opmath_t>(),
            scalar.to<opmath_t>());
      });
  increment_version(input);
}

template <template <class> class Op>
void foreach_pointwise_op_(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.reserve(3);
  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op__cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<3, opmath_t>(
            tensor_lists,
            scalars,
            PointwiseOpScalarListFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            Op<opmath_t>());
      });
  increment_version(input);
}

template <template <class> class Op>
std::vector<Tensor> foreach_pointwise_op(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.reserve(4);
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<4, opmath_t>(
            tensor_lists,
            scalars,
            PointwiseOpScalarListFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            Op<opmath_t>());
      });

  return std::move(tensor_lists[3]);
}

#define FOREACH_POINTWISE_OP_SCALAR(NAME, OP)                                 \
  std::vector<Tensor> foreach_tensor_##NAME##_scalar_cuda(                    \
      TensorList input,                                                       \
      TensorList tensors1,                                                    \
      TensorList tensors2,                                                    \
      const Scalar& scalar) {                                                 \
    check_foreach_api_restrictions(input, tensors1, tensors2);                \
                                                                              \
    if (has_integral_tensor(input, /* includeBool */ true)) {                 \
      return at::native::foreach_tensor_##NAME##_scalar_slow(                 \
          input, tensors1, tensors2, scalar);                                 \
    }                                                                         \
                                                                              \
    if (can_use_fast_route({input, tensors1, tensors2}, scalar)) {            \
      return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalar);     \
    }                                                                         \
                                                                              \
    /* Check if we can use 0D tensor1 fast path */                            \
    if (all_tensors_are_0d(tensors1) &&                                       \
        all_same_dtype(tensors1, input[0].scalar_type()) &&                   \
        all_same_device(tensors1, input[0].device()) &&                       \
        can_use_fast_route({input, tensors2}, scalar)) {                      \
      return foreach_pointwise_op_0d_tensor1<OP>(                             \
          input, tensors1, tensors2, scalar);                                 \
    }                                                                         \
    /* Check if we can use 0D tensor2 fast path (only for commutative ops) */ \
    if constexpr (!std::is_same_v<OP<float>, std::divides<float>>) {          \
      if (all_tensors_are_0d(tensors2) &&                                     \
          all_same_dtype(tensors2, input[0].scalar_type()) &&                 \
          all_same_device(tensors2, input[0].device()) &&                     \
          can_use_fast_route({input, tensors1}, scalar)) {                    \
        return foreach_pointwise_op_0d_tensor1<OP>(                           \
            input, tensors2, tensors1, scalar);                               \
      }                                                                       \
    }                                                                         \
                                                                              \
    return at::native::foreach_tensor_##NAME##_scalar_slow(                   \
        input, tensors1, tensors2, scalar);                                   \
  }                                                                           \
                                                                              \
  void foreach_tensor_##NAME##_scalar_cuda_(                                  \
      TensorList input,                                                       \
      TensorList tensors1,                                                    \
      TensorList tensors2,                                                    \
      const Scalar& scalar) {                                                 \
    check_foreach_api_restrictions(input, tensors1, tensors2);                \
                                                                              \
    if (has_integral_tensor(input, /* includeBool */ true)) {                 \
      return at::native::foreach_tensor_##NAME##_scalar_slow_(                \
          input, tensors1, tensors2, scalar);                                 \
    }                                                                         \
                                                                              \
    if (can_use_fast_route({input, tensors1, tensors2}, scalar)) {            \
      return foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalar);    \
    }                                                                         \
                                                                              \
    /* Check if we can use 0D tensor1 fast path */                            \
    if (all_tensors_are_0d(tensors1) &&                                       \
        all_same_dtype(tensors1, input[0].scalar_type()) &&                   \
        all_same_device(tensors1, input[0].device()) &&                       \
        can_use_fast_route({input, tensors2}, scalar)) {                      \
      return foreach_pointwise_op_0d_tensor1_<OP>(                            \
          input, tensors1, tensors2, scalar);                                 \
    }                                                                         \
    /* Check if we can use 0D tensor2 fast path (only for commutative ops) */ \
    if constexpr (!std::is_same_v<OP<float>, std::divides<float>>) {          \
      if (all_tensors_are_0d(tensors2) &&                                     \
          all_same_dtype(tensors2, input[0].scalar_type()) &&                 \
          all_same_device(tensors2, input[0].device()) &&                     \
          can_use_fast_route({input, tensors1}, scalar)) {                    \
        return foreach_pointwise_op_0d_tensor1_<OP>(                          \
            input, tensors2, tensors1, scalar);                               \
      }                                                                       \
    }                                                                         \
                                                                              \
    return at::native::foreach_tensor_##NAME##_scalar_slow_(                  \
        input, tensors1, tensors2, scalar);                                   \
  }

#define FOREACH_POINTWISE_OP_SCALARLIST(NAME, OP)                        \
  std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_cuda(           \
      TensorList input,                                                  \
      TensorList tensors1,                                               \
      TensorList tensors2,                                               \
      at::ArrayRef<Scalar> scalars) {                                    \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);  \
                                                                         \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalars) ||     \
        has_integral_tensor(input, /* includeBool */ true)) {            \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow(        \
          input, tensors1, tensors2, scalars);                           \
    }                                                                    \
                                                                         \
    return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalars); \
  }                                                                      \
                                                                         \
  void foreach_tensor_##NAME##_scalarlist_cuda_(                         \
      TensorList input,                                                  \
      TensorList tensors1,                                               \
      TensorList tensors2,                                               \
      at::ArrayRef<Scalar> scalars) {                                    \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);  \
                                                                         \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalars) ||     \
        has_integral_tensor(input, /* includeBool */ true)) {            \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow_(       \
          input, tensors1, tensors2, scalars);                           \
    }                                                                    \
                                                                         \
    foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalars);       \
  }

#define FOREACH_POINTWISE_OP_TENSOR(NAME, OP)                             \
  std::vector<Tensor> foreach_tensor_##NAME##_tensor_cuda(                \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Tensor& scalars_) {                                           \
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size()); \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);   \
    if (!can_use_fast_route({input, tensors1, tensors2}) ||               \
        has_integral_tensor(input, /* includeBool */ true)) {             \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow(         \
          input, tensors1, tensors2, scalars);                            \
    }                                                                     \
                                                                          \
    return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalars);  \
  }                                                                       \
                                                                          \
  void foreach_tensor_##NAME##_tensor_cuda_(                              \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Tensor& scalars_) {                                           \
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size()); \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);   \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalars) ||      \
        has_integral_tensor(input, /* includeBool */ true)) {             \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow_(        \
          input, tensors1, tensors2, scalars);                            \
    }                                                                     \
                                                                          \
    foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalars);        \
  }

FOREACH_POINTWISE_OP_SCALAR(addcmul, std::multiplies);
FOREACH_POINTWISE_OP_SCALAR(addcdiv, std::divides);
FOREACH_POINTWISE_OP_SCALARLIST(addcmul, std::multiplies);
FOREACH_POINTWISE_OP_SCALARLIST(addcdiv, std::divides);
FOREACH_POINTWISE_OP_TENSOR(addcdiv, std::divides);
FOREACH_POINTWISE_OP_TENSOR(addcmul, std::multiplies);

} // namespace at::native

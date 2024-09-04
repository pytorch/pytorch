#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/ForeachMinMaxFunctors.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_mul_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T, template <class> class Op>
std::vector<Tensor> foreach_binary_op(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  TORCH_CHECK(
      scalar.dim() == 0 && scalar.numel() == 1,
      "scalar tensor expected to be 0 dim but it has ",
      scalar.dim(),
      " dimensions and ",
      scalar.numel(),
      " elements.");
  TORCH_CHECK(
      tensors[0].device() == scalar.device(),
      "scalar tensor expected to be on ",
      tensors[0].device(),
      " but is on ",
      scalar.device());
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = at::opmath_type<T>;
  DISPATCH_MULTI_TENSOR_APPLY([&]() {
    multi_tensor_apply<2>(
        tensor_lists,
        BinaryOpScalarTensorFunctor<
            T,
            /* depth */ 2,
            /* r_args_depth */ 1,
            /* res_arg_index */ 1,
            large_kernel_arg>(),
        Op<opmath_t>(),
        scalar.data_ptr<T>(),
        alpha.to<opmath_t>());
  });
  return tensor_lists[1];
}

template <typename T, template <class> class Op>
void foreach_binary_op_(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  TORCH_CHECK(
      scalar.dim() == 0 && scalar.numel() == 1,
      "scalar tensor expected to be 0 dim but has ",
      scalar.dim(),
      " dimensions and ",
      scalar.numel(),
      " elements.");
  TORCH_CHECK(
      tensors[0].device() == scalar.device(),
      "scalar tensor is expected to be on ",
      tensors[0].device(),
      " but is on ",
      scalar.device());
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  using opmath_t = at::opmath_type<T>;
  DISPATCH_MULTI_TENSOR_APPLY([&]() {
    multi_tensor_apply<1>(
        tensor_lists,
        BinaryOpScalarTensorFunctor<
            T,
            /* depth */ 1,
            /* r_args_depth */ 1,
            /* res_arg_index */ 0,
            large_kernel_arg>(),
        Op<opmath_t>(),
        scalar.data_ptr<T>(),
        alpha.to<opmath_t>());
  });
  increment_version(tensors);
}

// TODO(crcrpar): Nest dispatch by looking up `scalar.scalar_type` for better
// coverage?
#define FOREACH_BINARY_OP_SCALAR_TENSOR(FUNCTION, NAME, OP, DIVISION_OP) \
  void foreach_tensor_##NAME##_tensor_kernel_cuda_(                      \
      TensorList tensors, const Tensor& scalar) {                        \
    if (scalar.device().type() == DeviceType::CPU) {                     \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_cuda_(    \
          tensors, scalar.item());                                       \
    }                                                                    \
    check_foreach_api_restrictions(tensors);                             \
    if (!(can_use_fast_route(                                            \
              ArrayRef<TensorList>{tensors}, {}, DIVISION_OP) &&         \
          tensors[0].scalar_type() == scalar.scalar_type())) {           \
      return at::native::foreach_tensor_##NAME##_tensor_kernel_slow_(    \
          tensors, scalar);                                              \
    }                                                                    \
                                                                         \
    FUNCTION##_<OP>(tensors, scalar);                                    \
  }                                                                      \
                                                                         \
  std::vector<Tensor> foreach_tensor_##NAME##_tensor_kernel_cuda(        \
      TensorList tensors, const Tensor& scalar) {                        \
    if (scalar.device().type() == DeviceType::CPU) {                     \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_cuda(     \
          tensors, scalar.item());                                       \
    }                                                                    \
    check_foreach_api_restrictions(tensors);                             \
    if (!(can_use_fast_route(                                            \
              ArrayRef<TensorList>{tensors}, {}, DIVISION_OP) &&         \
          tensors[0].scalar_type() == scalar.scalar_type())) {           \
      return at::native::foreach_tensor_##NAME##_tensor_kernel_slow(     \
          tensors, scalar);                                              \
    }                                                                    \
                                                                         \
    return FUNCTION<OP>(tensors, scalar);                                \
  }

#define FOREACH_BINARY_OP_SCALAR_TENSOR_ALPHA(FUNCTION, NAME, OP)      \
  void foreach_tensor_##NAME##_tensor_kernel_cuda_(                    \
      TensorList tensors, const Tensor& scalar, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors);                           \
    if (!(can_use_fast_route(ArrayRef<TensorList>{tensors}, alpha) &&  \
          tensors[0].scalar_type() == scalar.scalar_type())) {         \
      return at::native::foreach_tensor_##NAME##_tensor_kernel_slow_(  \
          tensors, scalar, alpha);                                     \
    }                                                                  \
                                                                       \
    FUNCTION##_<OP>(tensors, scalar, alpha);                           \
  }                                                                    \
                                                                       \
  std::vector<Tensor> foreach_tensor_##NAME##_tensor_kernel_cuda(      \
      TensorList tensors, const Tensor& scalar, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors);                           \
    if (!(can_use_fast_route(ArrayRef<TensorList>{tensors}, alpha) &&  \
          tensors[0].scalar_type() == scalar.scalar_type())) {         \
      return at::native::foreach_tensor_##NAME##_tensor_kernel_slow(   \
          tensors, scalar, alpha);                                     \
    }                                                                  \
                                                                       \
    return FUNCTION<OP>(tensors, scalar, alpha);                       \
  }

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() {
        return foreach_binary_op<scalar_t, Op>(tensors, scalar, alpha);
      });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar, alpha); });
}

FOREACH_BINARY_OP_SCALAR_TENSOR_ALPHA(
    all_types_complex_bool_half_bfloat16,
    add,
    std::plus);

FOREACH_BINARY_OP_SCALAR_TENSOR(
    all_types_complex_bool_half_bfloat16,
    mul,
    std::multiplies,
    /* div_op */ false);

FOREACH_BINARY_OP_SCALAR_TENSOR(
    all_types_complex_bool_half_bfloat16,
    div,
    std::divides,
    /* div_op */ true);

} // namespace at::native

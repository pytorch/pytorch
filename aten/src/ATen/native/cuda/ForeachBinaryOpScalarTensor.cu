#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/ForeachMinMaxFunctors.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_mul_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T, template <class> class Op>
std::vector<Tensor> foreach_binary_op(TensorList tensors, const Tensor& other) {
  TORCH_CHECK(
      other.dim() == 0 && other.numel() == 1,
      "other expected to be 0 dim but ",
      other.dim(),
      " and has ",
      other.numel(),
      " elements.");
  TORCH_CHECK(
      tensors[0].device() == other.device(),
      "other expected to be on ",
      tensors[0].device(),
      " but ",
      other.device());
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<2>(
      tensor_lists,
      BinaryOpScalarTensorFunctor<
          T,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),
      Op<opmath_t>(),
      other.data_ptr<opmath_t>());
  return tensor_lists[1];
}

template <typename T, template <class> class Op>
void foreach_binary_op_(TensorList tensors, const Tensor& other) {
  TORCH_CHECK(
      other.dim() == 0 && other.numel() == 1,
      "other expected to be 0 dim but ",
      other.dim(),
      " and has ",
      other.numel(),
      " elements.");
  TORCH_CHECK(
      tensors[0].device() == other.device(),
      "other expected to be on ",
      tensors[0].device(),
      " but ",
      other.device());
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<1>(
      tensor_lists,
      BinaryOpScalarTensorFunctor<
          T,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>(),
      other.data_ptr<opmath_t>());
  increment_version(tensors);
}

#define FOREACH_BINARY_OP_SCALAR_TENSOR(FUNCTION, NAME, OP, DIVISION_OP) \
  void foreach_tensor_##NAME##_tensor_kernel_cuda_(                      \
      TensorList tensors, const Tensor& scalar) {                        \
    check_foreach_api_restrictions(tensors);                             \
    if (!can_use_fast_route(tensors, scalar, DIVISION_OP)) {             \
      return at::native::foreach_tensor_##NAME##_tensor_kernel_slow_(    \
          tensors, scalar);                                              \
    }                                                                    \
                                                                         \
    FUNCTION##_<OP>(tensors, scalar);                                    \
  }                                                                      \
                                                                         \
  std::vector<Tensor> foreach_tensor_##NAME##_tensor_kernel_cuda(        \
      TensorList tensors, const Tensor& scalar) {                        \
    check_foreach_api_restrictions(tensors);                             \
    if (!can_use_fast_route(tensors, scalar, DIVISION_OP)) {             \
      return at::native::foreach_tensor_##NAME##_tensor_kernel_slow(     \
          tensors, scalar);                                              \
    }                                                                    \
                                                                         \
    return FUNCTION<OP>(tensors, scalar);                                \
  }

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors,
    const Tensor& scalar) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors,
    const Tensor& scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar); });
}

FOREACH_BINARY_OP_SCALAR_TENSOR(
    all_types_complex_bool_half_bfloat16,
    mul,
    std::multiplies,
    /* div_op */ false);
} // namespace at::native

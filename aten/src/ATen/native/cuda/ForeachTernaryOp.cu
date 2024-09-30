#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/Lerp.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_lerp_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T>
struct LerpFunctor {
  inline C10_DEVICE T operator()(const T self, const T end, const T weight) {
    return lerp(self, end, weight);
  }
};

std::vector<at::Tensor> foreach_tensor_lerp_ternary_cuda(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!can_use_fast_route({tensors1, tensors2, tensors3}, {}, true)) {
    return foreach_tensor_ternary_lerp_slow(tensors1, tensors2, tensors3);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), tensors3.vec(), vec_res};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_ternary_cuda",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<4>(
            tensor_lists,
            TernaryOpListFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            LerpFunctor<opmath_t>());
      });

  return tensor_lists[3];
}

void foreach_tensor_lerp_ternary_cuda_(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!can_use_fast_route({tensors1, tensors2, tensors3}, {}, true)) {
    return foreach_tensor_ternary_lerp_slow_(tensors1, tensors2, tensors3);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), tensors3.vec()};
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_ternary_cuda_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            TernaryOpListFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>());
      });
  increment_version(tensors1);
}

std::vector<at::Tensor> foreach_tensor_lerp_list_cuda(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2}, {}, true)) {
    return foreach_tensor_lerp_list_kernel_slow(tensors1, tensors2, weight);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), vec_res};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalar_cuda",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            TernaryOpScalarFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 2,
                /* res_arg_index */ 2>(),
            LerpFunctor<opmath_t>(),
            weight.to<opmath_t>());
      });

  return tensor_lists[2];
}

void foreach_tensor_lerp_list_cuda_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2}, {}, true)) {
    return foreach_tensor_lerp_list_kernel_slow_(tensors1, tensors2, weight);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec()};
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalar_cuda_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<2>(
            tensor_lists,
            TernaryOpScalarFunctor<
                scalar_t,
                /* depth */ 2,
                /* r_args_depth */ 2,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>(),
            weight.to<opmath_t>());
      });
}
} // namespace at::native

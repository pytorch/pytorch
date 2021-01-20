#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/BinaryOps.h>
namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(TensorList tensors, Scalar scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_binary_op_scalar_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2>(tensor_lists,
                              BinaryOpScalarFunctor<scalar_t, 
                                                    /* depth */ 2,
                                                    /* r_args_depth */ 1, 
                                                    /* res_arg_index */ 1>(),
                              Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });
    return tensor_lists[1];
}

template<template<class> class Op>
void foreach_binary_op_(TensorList tensors, Scalar scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_binary_op_scalar_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1>(tensor_lists,
                              BinaryOpScalarFunctor<scalar_t, 
                                                    /* depth */ 1,
                                                    /* r_args_depth */ 1, 
                                                    /* res_arg_index */ 0>(),
                                                    Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });
}

#define FOREACH_BINARY_OP_SCALAR(NAME, OP, DIVISION_OP)                                             \
void foreach_tensor_##NAME##_scalar_kernel_cuda_(TensorList tensors, Scalar scalar) {               \
    check_foreach_api_restrictions(tensors);                                                        \
    if (!can_use_fast_route(tensors, scalar, DIVISION_OP)) {                                        \
        return at::native::foreach_tensor_##NAME##_scalar_kernel_slow_(tensors, scalar);            \
    }                                                                                               \
                                                                                                    \
    foreach_binary_op_<OP>(tensors, scalar);                                                        \
}                                                                                                   \
                                                                                                    \
std::vector<Tensor> foreach_tensor_##NAME##_scalar_kernel_cuda(TensorList tensors, Scalar scalar) { \
    check_foreach_api_restrictions(tensors);                                                        \
    if (!can_use_fast_route(tensors, scalar, DIVISION_OP)) {                                        \
        return at::native::foreach_tensor_##NAME##_scalar_kernel_slow(tensors, scalar);             \
    }                                                                                               \
                                                                                                    \
    return foreach_binary_op<OP>(tensors, scalar);                                                  \
}

FOREACH_BINARY_OP_SCALAR(add, std::plus, false);
FOREACH_BINARY_OP_SCALAR(mul, std::multiplies, false);

// In the case of division, integer inputs will result in float. 
// Currently multi tensor apply can only return result of the same type as input.
FOREACH_BINARY_OP_SCALAR(div, std::divides, true);

// In the case of subtraction, we dont allow scalar to be boolean following the torch.sub logic
void foreach_tensor_sub_scalar_kernel_cuda_(TensorList tensors, Scalar scalar) {
    check_foreach_api_restrictions(tensors);
    at::native::sub_check(tensors[0], scalar);

    if (!can_use_fast_route(tensors, scalar)) {
        return at::native::foreach_tensor_sub_scalar_kernel_slow_(tensors, scalar);
    }

    foreach_binary_op_<std::minus>(tensors, scalar);
}

std::vector<Tensor> foreach_tensor_sub_scalar_kernel_cuda(TensorList tensors, Scalar scalar) {
    check_foreach_api_restrictions(tensors);
    at::native::sub_check(tensors[0], scalar);

    if (!can_use_fast_route(tensors, scalar)) {
        return at::native::foreach_tensor_sub_scalar_kernel_slow(tensors, scalar);
    }

    return foreach_binary_op<std::minus>(tensors, scalar);
}

}} // namespace at::native

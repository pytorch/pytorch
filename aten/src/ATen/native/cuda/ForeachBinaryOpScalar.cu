#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(TensorList tensors, Scalar scalar) {
    check_foreach_api_restrictions(tensors);

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_binary_op_scalar_cuda", [&]() {
        multi_tensor_apply<2>(tensor_lists, BinaryOpScalarFunctor<scalar_t, Op>(), scalar.to<scalar_t>());
    });
    return tensor_lists[1];
}

template<template<class> class Op>
void foreach_binary_op_(TensorList tensors, Scalar scalar) {
    check_foreach_api_restrictions(tensors);

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_binary_op_scalar_cuda_", [&]() {
        multi_tensor_apply<1>(tensor_lists, BinaryOpScalarFunctor_<scalar_t, Op>(), scalar.to<scalar_t>());
    });
}

#define FOREACH_BINARY_OP_SCALAR(NAME, OP)                                                          \
void foreach_tensor_##NAME##_scalar_kernel_cuda_(TensorList tensors, Scalar scalar) {               \
    check_foreach_api_restrictions(tensors);                                                        \
                                                                                                    \
    if (!can_use_fast_route(tensors, scalar)) {                                                     \
        return at::native::foreach_tensor_##NAME##_scalar_kernel_slow_(tensors, scalar);            \
    }                                                                                               \
                                                                                                    \
    foreach_binary_op_<OP>(tensors, scalar);                                                        \
}                                                                                                   \
                                                                                                    \
std::vector<Tensor> foreach_tensor_##NAME##_scalar_kernel_cuda(TensorList tensors, Scalar scalar) { \
    check_foreach_api_restrictions(tensors);                                                        \
                                                                                                    \
    if (!can_use_fast_route(tensors, scalar)) {                                                     \
        return at::native::foreach_tensor_##NAME##_scalar_kernel_slow(tensors, scalar);             \
    }                                                                                               \
                                                                                                    \
    return foreach_binary_op<OP>(tensors, scalar);                                                  \
}

FOREACH_BINARY_OP_SCALAR(add, std::plus);
FOREACH_BINARY_OP_SCALAR(sub, std::minus);
FOREACH_BINARY_OP_SCALAR(mul, std::multiplies);
FOREACH_BINARY_OP_SCALAR(div, std::divides);

}} // namespace at::native

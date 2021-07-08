#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/BinaryOps.h>

namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(TensorList tensors, at::ArrayRef<Scalar> scalars) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(vec_res);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBFloat16, kHalf, kBool, tensors[0].scalar_type(), "foreach_binary_op_scalarlist_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2, opmath_t>(tensor_lists,
                                        scalars,
                                        BinaryOpScalarListFunctor<scalar_t,
                                                                  /* depth */ 2,
                                                                  /* r_args_depth */ 1,
                                                                  /* res_arg_index */ 1>(),

                                        Op<opmath_t>());
    });
    return tensor_lists[1];
}

template<template<class> class Op>
void foreach_binary_op_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBFloat16, kHalf, kBool, tensors[0].scalar_type(), "foreach_binary_op_scalarlist_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1, opmath_t>(tensor_lists,
                                        scalars,
                                        BinaryOpScalarListFunctor<scalar_t,
                                                                  /* depth */ 1,
                                                                  /* r_args_depth */ 1,
                                                                  /* res_arg_index */ 0>(),
                                        Op<opmath_t>());
    });
}

#define FOREACH_BINARY_OP_SCALARLIST(NAME, OP, DIV_OP)                                                                   \
void foreach_tensor_##NAME##_scalarlist_kernel_cuda_(TensorList tensors, at::ArrayRef<Scalar> scalars) {                 \
    check_foreach_api_restrictions(tensors, scalars);                                                                    \
    if (!can_use_fast_route(tensors, scalars, DIV_OP)) {                                                                 \
        return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow_(tensors, scalars);                            \
    }                                                                                                                    \
                                                                                                                         \
    foreach_binary_op_<OP>(tensors, scalars);                                                                            \
}                                                                                                                        \
                                                                                                                         \
std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_kernel_cuda(TensorList tensors, at::ArrayRef<Scalar> scalars) {   \
    check_foreach_api_restrictions(tensors, scalars);                                                                    \
    if (!can_use_fast_route(tensors, scalars, DIV_OP)) {                                                                 \
        return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow(tensors, scalars);                             \
    }                                                                                                                    \
                                                                                                                         \
    return foreach_binary_op<OP>(tensors, scalars);                                                                      \
}

FOREACH_BINARY_OP_SCALARLIST(add, std::plus, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(mul, std::multiplies, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(div, std::divides, /*div_op*/ true);

// This does not use FOREACH_BINARY_OP_SCALARLIST because
// In the case of subtraction, we dont allow scalar to be boolean following the torch.sub logic
void foreach_tensor_sub_scalarlist_kernel_cuda_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
    check_foreach_api_restrictions(tensors, scalars);
    for (int i = 0; i < tensors.size(); i++) {
        sub_check(tensors[i], scalars[i]);
    }

    if (!can_use_fast_route({tensors}, scalars)) {
        return at::native::foreach_tensor_sub_scalarlist_kernel_slow_(tensors, scalars);
    }

    foreach_binary_op_<std::minus>(tensors, scalars);
}

std::vector<Tensor> foreach_tensor_sub_scalarlist_kernel_cuda(TensorList tensors, at::ArrayRef<Scalar> scalars) {
    check_foreach_api_restrictions(tensors, scalars);
    for (int i = 0; i < tensors.size(); i++) {
        sub_check(tensors[i], scalars[i]);
    }

    if (!can_use_fast_route({tensors}, scalars)) {
        return at::native::foreach_tensor_sub_scalarlist_kernel_slow(tensors, scalars);
    }

    return foreach_binary_op<std::minus>(tensors, scalars);
}

}} // namespace at::native

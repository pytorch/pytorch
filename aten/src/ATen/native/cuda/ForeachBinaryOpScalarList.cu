#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

template<typename scalar_t> struct get_compute_t;
template<> struct get_compute_t<at::Half> {
  using compute_t = float;
};
template<typename scalar_t> struct get_compute_t {
  using compute_t = scalar_t;
};

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(TensorList tensors, ScalarList scalars) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(vec_res);

    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_binary_op_scalarlist_cuda", [&]() {
        multi_tensor_apply2<2, scalar_t>(tensor_lists, scalars, BinaryOpScalarListFunctor<scalar_t, get_compute_t<scalar_t>::compute_t, Op>());
    });
    return tensor_lists[1];
}

template<template<class> class Op>
void foreach_binary_op_(TensorList tensors, ScalarList scalars) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_binary_op_scalarlist_cuda_", [&]() {
        multi_tensor_apply2<1, scalar_t>(tensor_lists, scalars, BinaryOpScalarListFunctor_<scalar_t, get_compute_t<scalar_t>::compute_t, Op>());
    });
}

//
// TODO: proper checks with scalar lists
// fast vs slow route

#define FOREACH_BINARY_OP_SCALARLIST(NAME, OP)                                                                 \
void foreach_tensor_##NAME##_scalarlist_kernel_cuda_(TensorList tensors, ScalarList scalars) {                 \
    check_foreach_api_restrictions(tensors, scalars);                                                          \
                                                                                                               \
    if (!can_use_fast_route(tensors, scalars)) {                                                               \
        return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow_(tensors, scalars);                  \
    }                                                                                                          \
                                                                                                               \
    foreach_binary_op_<OP>(tensors, scalars);                                                                  \
}                                                                                                              \
                                                                                                               \
std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_kernel_cuda(TensorList tensors, ScalarList scalars) {   \
    check_foreach_api_restrictions(tensors, scalars);                                                          \
                                                                                                               \
    if (!can_use_fast_route(tensors, scalars)) {                                                               \
        return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow(tensors, scalars);                   \
    }                                                                                                          \
                                                                                                               \
    return foreach_binary_op<OP>(tensors, scalars);                                                            \
}

FOREACH_BINARY_OP_SCALARLIST(add, std::plus);
FOREACH_BINARY_OP_SCALARLIST(sub, std::minus);
FOREACH_BINARY_OP_SCALARLIST(mul, std::multiplies);
FOREACH_BINARY_OP_SCALARLIST(div, std::divides);

}} // namespace at::native

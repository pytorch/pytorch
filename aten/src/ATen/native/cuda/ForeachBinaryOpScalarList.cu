#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(TensorList tensors, ScalarList scalars) {
    check_foreach_api_restrictions(tensors);

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(std::move(tensors.vec()));
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_binary_op_scalar_cuda", [&]() {
        multi_tensor_apply<2>(tensor_lists, BinaryOpScalarFunctor<scalar_t, Op>(), scalars[0].to<scalar_t>());
    });
    return tensor_lists[1];
}

template<template<class> class Op>
void foreach_binary_op_(TensorList tensors, ScalarList scalars) {
    check_foreach_api_restrictions(tensors);

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(std::move(tensors.vec()));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_binary_op_scalar_cuda_", [&]() {
        multi_tensor_apply2<1>(tensor_lists, scalars, BinaryOpScalarListFunctor_<scalar_t, Op>());
    });
}

void foreach_tensor_add_scalarlist_kernel_cuda_(TensorList tensors, ScalarList scalars) {                 
    check_foreach_api_restrictions(tensors);                                                                   
                                                                                                               
    if (!can_use_fast_route(tensors, scalars[0])) {                                                            
        return at::native::foreach_tensor_add_scalarlist_kernel_slow_(tensors, scalars[0]);                  
    }                                                                                                          
                                                                                                            
    foreach_binary_op_<std::plus>(tensors, scalars);                                                                  
}

std::vector<Tensor> foreach_tensor_add_scalarlist_kernel_cuda(TensorList tensors, ScalarList scalars) {   
    check_foreach_api_restrictions(tensors);                                                                   
                                                                                                               
    if (!can_use_fast_route(tensors, scalars[0])) {                                                            
        return at::native::foreach_tensor_add_scalarlist_kernel_slow(tensors, scalars[0]);                   
    }                                                                                                          
                                                                                                               
    return foreach_binary_op<std::plus>(tensors, scalars[0]);                                                            
}

// TODO: proper checks with scalar lists
#define FOREACH_BINARY_OP_SCALARLIST(NAME, OP)                                                                 \
void foreach_tensor_##NAME##_scalarlist_kernel_cuda_(TensorList tensors, ScalarList scalars) {                 \
    check_foreach_api_restrictions(tensors);                                                                   \
                                                                                                               \
    if (!can_use_fast_route(tensors, scalars[0])) {                                                            \
        return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow_(tensors, scalars[0]);                  \
    }                                                                                                          \
                                                                                                               \
    /*foreach_binary_op_<OP>(tensors, scalars[0]); */                                                                 \
}                                                                                                              \
                                                                                                               \
std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_kernel_cuda(TensorList tensors, ScalarList scalars) {   \
    check_foreach_api_restrictions(tensors);                                                                   \
                                                                                                               \
    if (!can_use_fast_route(tensors, scalars[0])) {                                                            \
        return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow(tensors, scalars[0]);                   \
    }                                                                                                          \
                                                                                                               \
    return foreach_binary_op<OP>(tensors, scalars[0]);                                                            \
}

//OREACH_BINARY_OP_SCALARLIST(add, std::plus);
FOREACH_BINARY_OP_SCALARLIST(sub, std::minus);
FOREACH_BINARY_OP_SCALARLIST(mul, std::multiplies);
FOREACH_BINARY_OP_SCALARLIST(div, std::divides);

}} // namespace at::native

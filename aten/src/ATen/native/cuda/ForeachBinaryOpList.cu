#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_tensor_list_op(TensorList tensors1, TensorList tensors2, Scalar alpha = 1) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    for (const auto& t: tensors1) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(std::move(tensors1.vec()));
    tensor_lists.emplace_back(std::move(tensors2.vec()));
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors1[0].scalar_type(), "foreach_binary_op_list_cuda", [&]() {
        multi_tensor_apply<3>(tensor_lists, BinaryOpListAlphaFunctor<scalar_t, Op>(), alpha.to<scalar_t>());
    });

    return tensor_lists[2];
}

template<template<class> class Op>
void foreach_tensor_list_op_(TensorList tensors1, TensorList tensors2, Scalar alpha = 1) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(std::move(tensors1.vec()));
    tensor_lists.emplace_back(std::move(tensors2.vec()));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors1[0].scalar_type(), "foreach_binary_op_list_cuda_", [&]() {
        multi_tensor_apply<2>(tensor_lists, BinaryOpListAlphaFunctor_<scalar_t, Op>(), alpha.to<scalar_t>());
    });
}

std::vector<Tensor> foreach_tensor_add_list_kernel_cuda(TensorList tensors1, TensorList tensors2, Scalar alpha) {
    check_foreach_api_restrictions(tensors1, tensors2);
    if (!can_use_fast_route(tensors1, tensors2)) {
        return at::native::foreach_tensor_add_list_kernel_slow(tensors1, tensors2, alpha);
    }

    return foreach_tensor_list_op<std::plus>(tensors1, tensors2, alpha);
}

void foreach_tensor_add_list_kernel_cuda_(TensorList tensors1, TensorList tensors2, Scalar alpha) {
    check_foreach_api_restrictions(tensors1, tensors2);

    if (!can_use_fast_route(tensors1, tensors2)) {
        return at::native::foreach_tensor_add_list_kernel_slow_(tensors1, tensors2, alpha);
    }

    foreach_tensor_list_op_<std::plus>(tensors1, tensors2, alpha);
}

std::vector<Tensor> foreach_tensor_sub_list_kernel_cuda(TensorList tensors1, TensorList tensors2, Scalar alpha) {
    check_foreach_api_restrictions(tensors1, tensors2);

    if (!can_use_fast_route(tensors1, tensors2)) {
        return at::native::foreach_tensor_sub_list_kernel_slow(tensors1, tensors2, alpha);
    }

    return foreach_tensor_list_op<std::minus>(tensors1, tensors2, alpha);
}

void foreach_tensor_sub_list_kernel_cuda_(TensorList tensors1, TensorList tensors2, Scalar alpha) {
    check_foreach_api_restrictions(tensors1, tensors2);

    if (!can_use_fast_route(tensors1, tensors2)) {
        return at::native::foreach_tensor_sub_list_kernel_slow_(tensors1, tensors2, alpha);
    }

    foreach_tensor_list_op_<std::minus>(tensors1, tensors2, alpha);
}

std::vector<Tensor> foreach_tensor_mul_list_kernel_cuda(TensorList tensors1, TensorList tensors2) {
    check_foreach_api_restrictions(tensors1, tensors2);

    if (!can_use_fast_route(tensors1, tensors2)) {
        return at::native::foreach_tensor_mul_list_kernel_slow(tensors1, tensors2);
    }

    return foreach_tensor_list_op<std::multiplies>(tensors1, tensors2);
}

void foreach_tensor_mul_list_kernel_cuda_(TensorList tensors1, TensorList tensors2) {
    check_foreach_api_restrictions(tensors1, tensors2);

    if (!can_use_fast_route(tensors1, tensors2)) {
        return at::native::foreach_tensor_mul_list_kernel_slow_(tensors1, tensors2);
    }

    foreach_tensor_list_op_<std::multiplies>(tensors1, tensors2);
}

std::vector<Tensor> foreach_tensor_div_list_kernel_cuda(TensorList tensors1, TensorList tensors2) {
    check_foreach_api_restrictions(tensors1, tensors2);

    if (!can_use_fast_route(tensors1, tensors2)) {
        return at::native::foreach_tensor_div_list_kernel_slow(tensors1, tensors2);
    }

    return foreach_tensor_list_op<std::divides>(tensors1, tensors2);
}

void foreach_tensor_div_list_kernel_cuda_(TensorList tensors1, TensorList tensors2) {
    check_foreach_api_restrictions(tensors1, tensors2);

    if (!can_use_fast_route(tensors1, tensors2)) {
        return at::native::foreach_tensor_div_list_kernel_slow_(tensors1, tensors2);
    }

    foreach_tensor_list_op_<std::divides>(tensors1, tensors2);
}

}} // namespace at::native

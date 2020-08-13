#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_pointwise_op(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    for (const auto& t: input) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(std::move(input.vec()));
    tensor_lists.emplace_back(std::move(tensors1.vec()));
    tensor_lists.emplace_back(std::move(tensors2.vec()));
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND(kHalf, input[0].scalar_type(), "foreach_pointwise_op_cuda", [&]() {
        multi_tensor_apply<4>(tensor_lists, PointwiseOpFunctor<scalar_t, Op>(), scalar.to<scalar_t>());
    });

    return tensor_lists[3];
}

template<template<class> class Op>
void foreach_pointwise_op_(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(std::move(input.vec()));
    tensor_lists.emplace_back(std::move(tensors1.vec()));
    tensor_lists.emplace_back(std::move(tensors2.vec()));

    AT_DISPATCH_ALL_TYPES_AND(kHalf, input[0].scalar_type(), "foreach_pointwise_op__cuda", [&]() {
        multi_tensor_apply<3>(tensor_lists, PointwiseOpFunctor_<scalar_t, Op>(), scalar.to<scalar_t>());
    });
}

std::vector<Tensor> foreach_tensor_addcdiv_cuda(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");
    TORCH_CHECK(input.size() ==  tensors1.size(), "Tensor lists must be of the same length.");
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

    if (!check_fast_route(input, scalar) ||
        !check_fast_route(tensors1, tensors2) ||
        !check_fast_route(input, tensors1)) {
        return at::native::foreach_tensor_addcdiv_slow(input, tensors1, tensors2, scalar);
    }

    return foreach_pointwise_op<std::divides>(input, tensors1, tensors2, scalar);
}

void foreach_tensor_addcdiv_cuda_(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");
    TORCH_CHECK(input.size() ==  tensors1.size(), "Tensor lists must be of the same length.");
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

    if (!check_fast_route(input, scalar) ||
        !check_fast_route(tensors1, tensors2) ||
        !check_fast_route(input, tensors1)) {
        at::native::foreach_tensor_addcdiv_slow_(input, tensors1, tensors2, scalar);
    }

    foreach_pointwise_op_<std::divides>(input, tensors1, tensors2, scalar);
}

std::vector<Tensor> foreach_tensor_addcmul_cuda(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");
    TORCH_CHECK(input.size() ==  tensors1.size(), "Tensor lists must be of the same length.");
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

    if (!check_fast_route(input, scalar) ||
        !check_fast_route(tensors1, tensors2) ||
        !check_fast_route(input, tensors1)) {
        return at::native::foreach_tensor_addcmul_slow(input, tensors1, tensors2, scalar);
    }

    return foreach_pointwise_op<std::multiplies>(input, tensors1, tensors2, scalar);
}

void foreach_tensor_addcmul_cuda_(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");
    TORCH_CHECK(input.size() ==  tensors1.size(), "Tensor lists must be of the same length.");
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

    if (!check_fast_route(input, scalar) ||
        !check_fast_route(tensors1, tensors2) ||
        !check_fast_route(input, tensors1)) {
        at::native::foreach_tensor_addcmul_slow_(input, tensors1, tensors2, scalar);
    }

    foreach_pointwise_op_<std::multiplies>(input, tensors1, tensors2, scalar);
}

}} // namespace at::native

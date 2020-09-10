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

#define FOREACH_UNARY_OP(NAME, OP)                                                                                             \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {  \
    TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");                                               \
    TORCH_CHECK(input.size() ==  tensors1.size(), "Tensor lists must be of the same length.");                                 \
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");                              \
                                                                                                                               \
    if (!can_use_fast_route(input, scalar) ||                                                                                  \
        !can_use_fast_route(tensors1, tensors2) ||                                                                             \
        !can_use_fast_route(input, tensors1)) {                                                                                \
        return at::native::foreach_tensor_##NAME##_slow(input, tensors1, tensors2, scalar);                                    \
    }                                                                                                                          \
                                                                                                                               \
    return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalar);                                                        \
}                                                                                                                              \
                                                                                                                               \
void foreach_tensor_##NAME##_cuda_(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {                \
    TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");                                               \
    TORCH_CHECK(input.size() ==  tensors1.size(), "Tensor lists must be of the same length.");                                 \
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");                              \
                                                                                                                               \
    if (!can_use_fast_route(input, scalar) ||                                                                                  \
        !can_use_fast_route(tensors1, tensors2) ||                                                                             \
        !can_use_fast_route(input, tensors1)) {                                                                                \
        at::native::foreach_tensor_##NAME##_slow_(input, tensors1, tensors2, scalar);                                          \
    }                                                                                                                          \
                                                                                                                               \
    foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalar);                                                              \
}

FOREACH_UNARY_OP(addcmul, std::multiplies);
FOREACH_UNARY_OP(addcdiv, std::divides);

}} // namespace at::native

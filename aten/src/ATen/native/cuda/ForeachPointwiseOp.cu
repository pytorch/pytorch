#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/NumericUtils.h>

namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_pointwise_op(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(input.size());
    for (const auto& t: input) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(input.vec());
    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND(kHalf, input[0].scalar_type(), "foreach_pointwise_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<4>(tensor_lists,
                              PointwiseOpScalarFunctor<scalar_t, 
                                                       /* depth */ 4,
                                                       /* r_args_depth */ 3, 
                                                       /* res_arg_index */ 3>(),
                              Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });

    return tensor_lists[3];
}

template<template<class> class Op>
void foreach_pointwise_op_(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(input.vec());
    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());

    AT_DISPATCH_ALL_TYPES_AND(kHalf, input[0].scalar_type(), "foreach_pointwise_op__cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<3>(tensor_lists,
                              PointwiseOpScalarFunctor<scalar_t, 
                                                       /* depth */ 3,
                                                       /* r_args_depth */ 3, 
                                                       /* res_arg_index */ 0>(),
                              Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });
}

template<template<class> class Op>
void foreach_pointwise_op_(TensorList input, TensorList tensors1, TensorList tensors2, at::ArrayRef<Scalar> scalars) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.reserve(3);
    tensor_lists.emplace_back(input.vec());
    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());

    AT_DISPATCH_ALL_TYPES_AND(kHalf, input[0].scalar_type(), "foreach_pointwise_op__cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<3, opmath_t>(tensor_lists,
                                        scalars,
                                        PointwiseOpScalarListFunctor<scalar_t, 
                                                                     /* depth */ 3,
                                                                     /* r_args_depth */ 3, 
                                                                     /* res_arg_index */ 0>(),
                                        Op<opmath_t>());
    });
}

template<template<class> class Op>
std::vector<Tensor> foreach_pointwise_op(TensorList input, TensorList tensors1, TensorList tensors2, at::ArrayRef<Scalar> scalars) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.reserve(4);
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(input.size());
    for (const auto& t: input) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(input.vec());
    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND(kHalf, input[0].scalar_type(), "foreach_pointwise_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<4, opmath_t>(tensor_lists,
                                        scalars,
                                        PointwiseOpScalarListFunctor<scalar_t, 
                                                                     /* depth */ 4,
                                                                     /* r_args_depth */ 3, 
                                                                     /* res_arg_index */ 3>(),
                                        Op<opmath_t>());
    });

    return tensor_lists[3];
}

#define FOREACH_POINTWISE_OP_SCALAR(NAME, OP)                                                                                         \
std::vector<Tensor> foreach_tensor_##NAME##_scalar_cuda(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {  \
    check_foreach_api_restrictions(input, tensors1, tensors2);                                                                        \
                                                                                                                                      \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalar)) {                                                                   \
        return at::native::foreach_tensor_##NAME##_scalar_slow(input, tensors1, tensors2, scalar);                                    \
    }                                                                                                                                 \
                                                                                                                                      \
    return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalar);                                                               \
}                                                                                                                                     \
                                                                                                                                      \
void foreach_tensor_##NAME##_scalar_cuda_(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {                \
    check_foreach_api_restrictions(input, tensors1, tensors2);                                                                        \
                                                                                                                                      \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalar)) {                                                                   \
        return at::native::foreach_tensor_##NAME##_scalar_slow_(input, tensors1, tensors2, scalar);                                   \
    }                                                                                                                                 \
                                                                                                                                      \
    foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalar);                                                                     \
}


#define FOREACH_POINTWISE_OP_SCALARLIST(NAME, OP)                                                                                                        \
std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_cuda(TensorList input, TensorList tensors1, TensorList tensors2, at::ArrayRef<Scalar> scalars) {  \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);                                                                                  \
                                                                                                                                                         \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalars)) {                                                                                     \
        return at::native::foreach_tensor_##NAME##_scalarlist_slow(input, tensors1, tensors2, scalars);                                                  \
    }                                                                                                                                                    \
                                                                                                                                                         \
    return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalars);                                                                                 \
}                                                                                                                                                        \
                                                                                                                                                         \
void foreach_tensor_##NAME##_scalarlist_cuda_(TensorList input, TensorList tensors1, TensorList tensors2, at::ArrayRef<Scalar> scalars) {                \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);                                                                                  \
                                                                                                                                                         \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalars)) {                                                                                     \
        return at::native::foreach_tensor_##NAME##_scalarlist_slow_(input, tensors1, tensors2, scalars);                                                 \
    }                                                                                                                                                    \
                                                                                                                                                         \
    foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalars);                                                                                       \
}

FOREACH_POINTWISE_OP_SCALAR(addcmul, std::multiplies);
FOREACH_POINTWISE_OP_SCALAR(addcdiv, std::divides);
FOREACH_POINTWISE_OP_SCALARLIST(addcmul, std::multiplies);
FOREACH_POINTWISE_OP_SCALARLIST(addcdiv, std::divides);

#define FOREACH_MAXIMUM_MINIMUM_OP(NAME, OP)                                                               \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList tensors1, TensorList tensors2) {               \
    check_foreach_api_restrictions(tensors1, tensors2);                                                    \
    if (!can_use_fast_route({tensors1, tensors2})) {                                                       \
        return at::native::foreach_tensor_##NAME##_slow(tensors1, tensors2);                               \
    }                                                                                                      \
                                                                                                           \
    std::vector<std::vector<at::Tensor>> tensor_lists;                                                     \
    std::vector<at::Tensor> vec_res;                                                                       \
    vec_res.reserve(tensors1.size());                                                                      \
    for (const auto& t: tensors1) {                                                                        \
        vec_res.emplace_back(at::native::empty_like(t));                                                   \
    }                                                                                                      \
                                                                                                           \
    tensor_lists.emplace_back(tensors1.vec());                                                             \
    tensor_lists.emplace_back(tensors2.vec());                                                             \
    tensor_lists.emplace_back(std::move(vec_res));                                                         \
                                                                                                           \
    AT_DISPATCH_ALL_TYPES_AND(kHalf, tensors1[0].scalar_type(), "foreach_maximum_minimum_op_cuda", [&]() { \
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;                                                 \
        auto op = []  GPU_LAMBDA (opmath_t a, opmath_t b) -> opmath_t {                                    \
            opmath_t c = a OP b ? a : b;                                                                   \
            if (_isnan(a)) {                                                                               \
              c = a;                                                                                       \
            }                                                                                              \
            return c;};                                                                                    \
        multi_tensor_apply<3>(tensor_lists,                                                                \
                              PointwiseOpListFunctor<scalar_t, 3>(),                                       \
                              op);                                                                         \
    });                                                                                                    \
                                                                                                           \
    return tensor_lists[2];                                                                                \
}                                                                                                          \

FOREACH_MAXIMUM_MINIMUM_OP(maximum, >)
FOREACH_MAXIMUM_MINIMUM_OP(minimum, <)

}} // namespace at::native

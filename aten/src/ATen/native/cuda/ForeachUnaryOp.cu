#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

template <template<class> class Op>
std::vector<Tensor> foreach_unary_op(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half,  tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 2,
                                             /* r_args_depth */ 1, 
                                             /* res_arg_index */ 1>(),
                              Op<opmath_t>());
    });
    return tensor_lists[1];
}

template <template<class> class Op>
void foreach_unary_op_(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 1,
                                             /* r_args_depth */ 1, 
                                             /* res_arg_index */ 0>(),
                              Op<opmath_t>());
    });
}

template <template<class> class Op>
std::vector<Tensor> foreach_unary_op_no_comlex(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Half,  tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 2,
                                             /* r_args_depth */ 1, 
                                             /* res_arg_index */ 1>(),
                              Op<opmath_t>());
    });
    return tensor_lists[1];
}

template <template<class> class Op>
void foreach_unary_op_no_comlex_(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Half, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 1,
                                             /* r_args_depth */ 1, 
                                             /* res_arg_index */ 0>(),
                              Op<opmath_t>());
    });
}

#define FOREACH_UNARY_OP(NAME, NAME1)                                   \
template<typename T>                                                    \
struct NAME1 {                                                          \
    __device__ T operator()(T t) const { return std::NAME(t); }         \
};                                                                      \
                                                                        \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList tensors) {  \
    check_foreach_api_restrictions(tensors);                            \
                                                                        \
    if (!can_use_fast_route(tensors)) {                                 \
        return at::native::foreach_tensor_##NAME##_slow(tensors);       \
    }                                                                   \
                                                                        \
    return foreach_unary_op<NAME1>(tensors);                            \
}                                                                       \
                                                                        \
void foreach_tensor_##NAME##_cuda_(TensorList tensors) {                \
    check_foreach_api_restrictions(tensors);                            \
                                                                        \
    if (!can_use_fast_route(tensors)) {                                 \
        return at::native::foreach_tensor_##NAME##_slow_(tensors);      \
    }                                                                   \
                                                                        \
    foreach_unary_op_<NAME1>(tensors);                                  \
}

#define FOREACH_UNARY_OP_NO_COMPLEX(NAME, NAME1)                        \
template<typename T>                                                    \
struct NAME1 {                                                          \
    __device__ T operator()(T t) const { return std::NAME(t); }         \
};                                                                      \
                                                                        \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList tensors) {  \
    check_foreach_api_restrictions(tensors);                            \
                                                                        \
    if (!can_use_fast_route(tensors)) {                                 \
        return at::native::foreach_tensor_##NAME##_slow(tensors);       \
    }                                                                   \
                                                                        \
    return foreach_unary_op_no_comlex<NAME1>(tensors);                  \
}                                                                       \
                                                                        \
void foreach_tensor_##NAME##_cuda_(TensorList tensors) {                \
    check_foreach_api_restrictions(tensors);                            \
                                                                        \
    if (!can_use_fast_route(tensors)) {                                 \
        return at::native::foreach_tensor_##NAME##_slow_(tensors);      \
    }                                                                   \
                                                                        \
    foreach_unary_op_no_comlex_<NAME1>(tensors);                        \
}

FOREACH_UNARY_OP(exp, Exp);
FOREACH_UNARY_OP(sqrt, Sqrt);
FOREACH_UNARY_OP(acos, Acos);
FOREACH_UNARY_OP(asin, Asin);
FOREACH_UNARY_OP(atan, Atan);
FOREACH_UNARY_OP(cos, Cos);
FOREACH_UNARY_OP(cosh, Cosh);
FOREACH_UNARY_OP(log, Log);
FOREACH_UNARY_OP(log10, Log10);
FOREACH_UNARY_OP(log2, Log2);
FOREACH_UNARY_OP(tan, Tan);
FOREACH_UNARY_OP(tanh, Tanh);
FOREACH_UNARY_OP(sin, Sin);
FOREACH_UNARY_OP(sinh, Sinh);

FOREACH_UNARY_OP_NO_COMPLEX(abs, Abs);
FOREACH_UNARY_OP_NO_COMPLEX(ceil, Ceil);
FOREACH_UNARY_OP_NO_COMPLEX(erf, Erf);
FOREACH_UNARY_OP_NO_COMPLEX(erfc, Erfc);
FOREACH_UNARY_OP_NO_COMPLEX(log1p, Log1p);
FOREACH_UNARY_OP_NO_COMPLEX(expm1, Expm1);
FOREACH_UNARY_OP_NO_COMPLEX(floor, Floor);
FOREACH_UNARY_OP_NO_COMPLEX(round, Round);
FOREACH_UNARY_OP_NO_COMPLEX(lgamma, Lgamma);

std::vector<Tensor> foreach_tensor_neg_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_neg_slow(tensors);
    }

    return foreach_unary_op<std::negate>(tensors);
}

void foreach_tensor_neg_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_neg_slow_(tensors);
    }

    foreach_unary_op_<std::negate>(tensors);
}

}} // namespace at::native

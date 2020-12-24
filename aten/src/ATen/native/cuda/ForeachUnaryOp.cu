#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

template <template<class> class Op>
std::vector<Tensor> foreach_unary_op_complex(TensorList tensors) {
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
void foreach_unary_op_complex_(TensorList tensors) {
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
std::vector<Tensor> foreach_unary_op_complex_bfloat16(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
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
void foreach_unary_op_complex_bfloat16_(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
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
std::vector<Tensor> foreach_unary_op(TensorList tensors) {
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
void foreach_unary_op_(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
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
void foreach_op_unary_(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
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
std::vector<Tensor> foreach_unary_op_bfloat16(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16,  tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
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
void foreach_unary_op_bfloat16_(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 1,
                                             /* r_args_depth */ 1, 
                                             /* res_arg_index */ 0>(),
                              Op<opmath_t>());
    });
}

#define FOREACH_UNARY_OP_COMPLEX(NAME, NAME1)                           \
template<typename T>                                                    \
struct NAME1 {                                                          \
    __device__ T operator()(T t) const { return std::NAME(t); }         \
};                                                                      \
                                                                        \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList tensors) {  \
    check_foreach_api_restrictions(tensors);                            \
    if (!can_use_fast_route(tensors)) {                                 \
        return at::native::foreach_tensor_##NAME##_slow(tensors);       \
    }                                                                   \
                                                                        \
    return foreach_unary_op_complex<NAME1>(tensors);                    \
}                                                                       \
                                                                        \
void foreach_tensor_##NAME##_cuda_(TensorList tensors) {                \
    check_foreach_api_restrictions(tensors);                            \
    if (!can_use_fast_route(tensors)) {                                 \
        return at::native::foreach_tensor_##NAME##_slow_(tensors);      \
    }                                                                   \
                                                                        \
    foreach_unary_op_complex_<NAME1>(tensors);                          \
}

#define FOREACH_UNARY_OP_COMPLEX_BFLOAT16(NAME, NAME1)                  \
template<typename T>                                                    \
struct NAME1 {                                                          \
    __device__ T operator()(T t) const { return std::NAME(t); }         \
};                                                                      \
                                                                        \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList tensors) {  \
    check_foreach_api_restrictions(tensors);                            \
    if (!can_use_fast_route(tensors)) {                                 \
        return at::native::foreach_tensor_##NAME##_slow(tensors);       \
    }                                                                   \
                                                                        \
    return foreach_unary_op_complex_bfloat16<NAME1>(tensors);           \
}                                                                       \
                                                                        \
void foreach_tensor_##NAME##_cuda_(TensorList tensors) {                \
    check_foreach_api_restrictions(tensors);                            \
    if (!can_use_fast_route(tensors)) {                                 \
        return at::native::foreach_tensor_##NAME##_slow_(tensors);      \
    }                                                                   \
                                                                        \
    foreach_unary_op_complex_bfloat16_<NAME1>(tensors);                 \
}

#define FOREACH_UNARY_OP(NAME, NAME1)                                   \
template<typename T>                                                    \
struct NAME1 {                                                          \
    __device__ T operator()(T t) const { return std::NAME(t); }         \
};                                                                      \
                                                                        \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList tensors) {  \
    check_foreach_api_restrictions(tensors);                            \
    if (!can_use_fast_route(tensors)) {                                 \
        return at::native::foreach_tensor_##NAME##_slow(tensors);       \
    }                                                                   \
                                                                        \
    return foreach_unary_op<NAME1>(tensors);                            \
}                                                                       \
                                                                        \
void foreach_tensor_##NAME##_cuda_(TensorList tensors) {                \
    check_foreach_api_restrictions(tensors);                            \
    if (!can_use_fast_route(tensors)) {                                 \
        return at::native::foreach_tensor_##NAME##_slow_(tensors);      \
    }                                                                   \
                                                                        \
    foreach_unary_op_<NAME1>(tensors);                                  \
}

#define FOREACH_UNARY_OP_BFLOAT16(NAME, NAME1)                          \
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
    return foreach_unary_op_bfloat16<NAME1>(tensors);                   \
}                                                                       \
                                                                        \
void foreach_tensor_##NAME##_cuda_(TensorList tensors) {                \
    check_foreach_api_restrictions(tensors);                            \
                                                                        \
    if (!can_use_fast_route(tensors)) {                                 \
        return at::native::foreach_tensor_##NAME##_slow_(tensors);      \
    }                                                                   \
                                                                        \
    foreach_unary_op_bfloat16_<NAME1>(tensors);                         \
}

FOREACH_UNARY_OP(ceil, Ceil);
FOREACH_UNARY_OP(erfc, Erfc);
FOREACH_UNARY_OP(expm1, Expm1);
FOREACH_UNARY_OP(floor, Floor);
FOREACH_UNARY_OP(lgamma, Lgamma);

FOREACH_UNARY_OP_BFLOAT16(log1p, Log1p);
FOREACH_UNARY_OP_BFLOAT16(erf, Erf);

FOREACH_UNARY_OP_COMPLEX(acos, Acos);
FOREACH_UNARY_OP_COMPLEX(asin, Asin);
FOREACH_UNARY_OP_COMPLEX(atan, Atan);
FOREACH_UNARY_OP_COMPLEX(cosh, Cosh);
FOREACH_UNARY_OP_COMPLEX(tan, Tan);
FOREACH_UNARY_OP_COMPLEX(sin, Sin);
FOREACH_UNARY_OP_COMPLEX(sinh, Sinh);

FOREACH_UNARY_OP_COMPLEX_BFLOAT16(exp, Exp);
FOREACH_UNARY_OP_COMPLEX_BFLOAT16(sqrt, Sqrt);
FOREACH_UNARY_OP_COMPLEX_BFLOAT16(cos, Cos);
FOREACH_UNARY_OP_COMPLEX_BFLOAT16(tanh, Tanh);
FOREACH_UNARY_OP_COMPLEX_BFLOAT16(log, Log);
FOREACH_UNARY_OP_COMPLEX_BFLOAT16(log10, Log10);
FOREACH_UNARY_OP_COMPLEX_BFLOAT16(log2, Log2);

//
// Special cases
//
std::vector<Tensor> foreach_tensor_neg_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_neg_slow(tensors);
    }

    return foreach_unary_op_complex_bfloat16<std::negate>(tensors);
}

void foreach_tensor_neg_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_neg_slow_(tensors);
    }

    foreach_unary_op_complex_bfloat16_<std::negate>(tensors);
}

template<typename T>                                                    \
struct Round {                                                          \
    __device__ T operator()(T t) const { return std::nearbyint(t); }    \
};

std::vector<Tensor> foreach_tensor_round_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_round_slow(tensors);
    }

    return foreach_unary_op<Round>(tensors);
}

void foreach_tensor_round_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_round_slow_(tensors);
    }

    foreach_unary_op_<Round>(tensors);
}

// Abs have to go via slow path in case of a complex type.
// This is because foreach kernels can't return a different dtype than passed, while 
// abs with complex input will produce float output.
template<typename T>
struct Abs {
    __device__ T operator()(T t) const { return std::abs(t); }
};

std::vector<Tensor> foreach_tensor_abs_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);
    bool has_complex = false;
    for (auto t : tensors) {
        if (at::isComplexType(t.scalar_type())) {
            has_complex = true;
        }
    }

    if (!can_use_fast_route(tensors) || has_complex) {
        return at::native::foreach_tensor_abs_slow(tensors);
    }

    return foreach_unary_op_complex_bfloat16<Abs>(tensors);
}

void foreach_tensor_abs_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);
    bool has_complex = false;
    for (auto t : tensors) {
        if (at::isComplexType(t.scalar_type())) {
            has_complex = true;
        }
    }

    if (!can_use_fast_route(tensors) || has_complex) {
        return at::native::foreach_tensor_abs_slow_(tensors);
    }

    foreach_unary_op_complex_bfloat16_<Abs>(tensors);
}

template<typename T>
struct Trunc {
    __device__ T operator()(T t) const { return t - std::trunc(t); }
};

std::vector<Tensor> foreach_tensor_frac_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_frac_slow(tensors);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 2,
                                             /* r_args_depth */ 1,
                                             /* res_arg_index */ 1>(),
                              Trunc<opmath_t>());
    });
    return tensor_lists[1];
}

void foreach_tensor_frac_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_frac_slow_(tensors);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 1,
                                             /* r_args_depth */ 1,
                                             /* res_arg_index */ 0>(),
                              Trunc<opmath_t>());
    });
}

template<typename T>
struct Sigmoid {
    T one = T(1);
    __device__ T operator()(T t) const { return (one / (one + std::exp(-t))); }
};

std::vector<Tensor> foreach_tensor_sigmoid_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_sigmoid_slow(tensors);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 2,
                                             /* r_args_depth */ 1, 
                                             /* res_arg_index */ 1>(),
                              Sigmoid<opmath_t>());
    });
    return tensor_lists[1];
}

void foreach_tensor_sigmoid_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_sigmoid_slow_(tensors);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 1,
                                             /* r_args_depth */ 1,
                                             /* res_arg_index */ 0>(),
                              Sigmoid<opmath_t>());
    });
}

template<typename T>
struct Reciprocal {
    T one = T(1);
    __device__ T operator()(T t) const { return (one / t); }
};

std::vector<Tensor> foreach_tensor_reciprocal_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_reciprocal_slow(tensors);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 2,
                                             /* r_args_depth */ 1, 
                                             /* res_arg_index */ 1>(),
                              Reciprocal<opmath_t>());
    });
    return tensor_lists[1];
}

void foreach_tensor_reciprocal_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_reciprocal_slow_(tensors);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 1,
                                             /* r_args_depth */ 1, 
                                             /* res_arg_index */ 0>(),
                              Reciprocal<opmath_t>());
    });
}

template<typename T>
struct Truncf {
    __device__ T operator()(T t) const { return std::trunc(t); }
};

std::vector<Tensor> foreach_tensor_trunc_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_trunc_slow(tensors);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 2,
                                             /* r_args_depth */ 1, 
                                             /* res_arg_index */ 1>(),
                              Truncf<opmath_t>());
    });
    return tensor_lists[1];
}

void foreach_tensor_trunc_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_trunc_slow_(tensors);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1>(tensor_lists,
                              UnaryOpFunctor<scalar_t,
                                             /* depth */ 1,
                                             /* r_args_depth */ 1, 
                                             /* res_arg_index */ 0>(),
                              Truncf<opmath_t>());
    });
}

void foreach_tensor_zero_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_zero_slow_(tensors);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, tensors[0].scalar_type(), "foreach_zero_cuda_", [&]() {
        multi_tensor_apply<1>(tensor_lists,
                              ZeroFunctor<scalar_t,
                                          /* depth */ 1,
                                          /* r_args_depth */ 1, 
                                          /* res_arg_index */ 0>());
    });
}

}} // namespace at::native

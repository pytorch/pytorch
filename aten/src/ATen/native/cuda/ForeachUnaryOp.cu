#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {
template <template<class> class Op>
std::vector<Tensor> floating_complex_half(TensorList tensors) {
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
void floating_complex_half_(TensorList tensors) {
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
std::vector<Tensor> floating_complex_half_bfloat16(TensorList tensors) {
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
void floating_complex_half_bfloat16_(TensorList tensors) {
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
std::vector<Tensor> all_types_half_bfloat16(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, at::ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
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
void all_types_half_bfloat16_(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, at::ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
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
std::vector<Tensor> floating_half(TensorList tensors) {
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
void floating_half_(TensorList tensors) {
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
std::vector<Tensor> floating_half_bfloat16(TensorList tensors) {
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
void floating_half_bfloat16_(TensorList tensors) {
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

#define FLOATING_COMPLEX_HALF(NAME, NAME1)                               \
template<typename T>                                                     \
struct NAME1 {                                                           \
    __device__ T operator()(T t) const { return std::NAME(t); }          \
};                                                                       \
                                                                         \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList tensors) {   \
    check_foreach_api_restrictions(tensors);                             \
    bool has_integral = has_int_or_bool_tensor(tensors);                 \
        /* MTA doesnt support different return type than input one */    \
        if (!can_use_fast_route(tensors) || has_integral) {              \
            return at::native::foreach_tensor_##NAME##_slow(tensors);    \
        }                                                                \
                                                                         \
    return floating_complex_half<NAME1>(tensors);                        \
}                                                                        \
                                                                         \
void foreach_tensor_##NAME##_cuda_(TensorList tensors) {                 \
    bool has_integral = has_int_or_bool_tensor(tensors);                 \
    /* MTA doesnt support different return type than input one */        \
    if (!can_use_fast_route(tensors) || has_integral) {                  \
        return at::native::foreach_tensor_##NAME##_slow_(tensors);       \
    }                                                                    \
                                                                         \
    floating_complex_half_<NAME1>(tensors);                              \
}

#define FLOATING_COMPLEX_HALF_BFLOAT16(NAME, NAME1)                      \
template<typename T>                                                     \
struct NAME1 {                                                           \
    __device__ T operator()(T t) const { return std::NAME(t); }          \
};                                                                       \
                                                                         \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList tensors) {   \
    check_foreach_api_restrictions(tensors);                             \
    bool has_integral = has_int_or_bool_tensor(tensors);                 \
    /* MTA doesnt support different return type than input one */        \
    if (!can_use_fast_route(tensors) || has_integral) {                  \
        return at::native::foreach_tensor_##NAME##_slow(tensors);        \
    }                                                                    \
    return floating_complex_half_bfloat16<NAME1>(tensors);               \
}                                                                        \
                                                                         \
void foreach_tensor_##NAME##_cuda_(TensorList tensors) {                 \
    check_foreach_api_restrictions(tensors);                             \
    bool has_integral = has_int_or_bool_tensor(tensors);                 \
    /* MTA doesnt support different return type than input one */        \
    if (!can_use_fast_route(tensors) || has_integral) {                  \
        return at::native::foreach_tensor_##NAME##_slow_(tensors);       \
    }                                                                    \
    floating_complex_half_bfloat16_<NAME1>(tensors);                     \
}

#define FLOATING_HALF_BFLOAT16(NAME, NAME1)                             \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList tensors) {  \
    check_foreach_api_restrictions(tensors);                            \
    bool has_integral = has_int_or_bool_tensor(tensors);                \
    /* MTA doesnt support different return type than input one */       \
    if (!can_use_fast_route(tensors) || has_integral) {                 \
        return at::native::foreach_tensor_##NAME##_slow(tensors);       \
    }                                                                   \
    return floating_half_bfloat16<NAME1>(tensors);                      \
}                                                                       \
                                                                        \
void foreach_tensor_##NAME##_cuda_(TensorList tensors) {                \
    check_foreach_api_restrictions(tensors);                            \
    bool has_integral = has_int_or_bool_tensor(tensors);                \
    /* MTA doesnt support different return type than input one */       \
    if (!can_use_fast_route(tensors) || has_integral) {                 \
        return at::native::foreach_tensor_##NAME##_slow_(tensors);      \
    }                                                                   \
                                                                        \
    floating_half_bfloat16_<NAME1>(tensors);                            \
}

#define FLOATING_HALF(NAME, NAME1, SUPPORTS_COMPLEX, SUPPORTS_INT)                 \
template<typename T>                                                               \
struct NAME1 {                                                                     \
    __device__ T operator()(T t) const { return std::NAME(t); }                    \
};                                                                                 \
                                                                                   \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList tensors) {             \
    check_foreach_api_restrictions(tensors);                                       \
    if (!SUPPORTS_COMPLEX) {                                                       \
        TORCH_CHECK(!tensors[0].is_complex(), "Not supported for complex inputs"); \
    }                                                                              \
                                                                                   \
    if (!SUPPORTS_INT) {                                                           \
        bool has_integral = has_int_or_bool_tensor(tensors);                       \
        /* MTA doesnt support different return type than input one */              \
        if (!can_use_fast_route(tensors) || has_integral) {                        \
            return at::native::foreach_tensor_##NAME##_slow(tensors);              \
        }                                                                          \
    }                                                                              \
    else {                                                                         \
        if (!can_use_fast_route(tensors)) {                                        \
            return at::native::foreach_tensor_##NAME##_slow(tensors);              \
        }                                                                          \
    }                                                                              \
                                                                                   \
    return floating_half<NAME1>(tensors);                                          \
}                                                                                  \
                                                                                   \
void foreach_tensor_##NAME##_cuda_(TensorList tensors) {                           \
    check_foreach_api_restrictions(tensors);                                       \
    if (!SUPPORTS_COMPLEX) {                                                       \
        TORCH_CHECK(!tensors[0].is_complex(), "Not supported for complex inputs"); \
    }                                                                              \
                                                                                   \
    if (!SUPPORTS_INT) {                                                           \
        bool has_integral = has_int_or_bool_tensor(tensors);                       \
        /* MTA doesnt support different return type than input one */              \
        if (!can_use_fast_route(tensors) || has_integral) {                        \
            return at::native::foreach_tensor_##NAME##_slow_(tensors);             \
        }                                                                          \
    }                                                                              \
    else {                                                                         \
        if (!can_use_fast_route(tensors)) {                                        \
            return at::native::foreach_tensor_##NAME##_slow_(tensors);             \
        }                                                                          \
    }                                                                              \
                                                                                   \
    floating_half_<NAME1>(tensors);                                                \
}

FLOATING_HALF(erfc, Erfc, true, false);
FLOATING_HALF(expm1, Expm1, true, false);
FLOATING_HALF(lgamma, Lgamma, true, false);
FLOATING_HALF(trunc, Truncf, false, true);
FLOATING_HALF(floor, Floor, false, true);
FLOATING_HALF(ceil, Ceil, false, true);

template<typename T>
struct Log1p {
    __device__ T operator()(T t) const { return std::log1p(t); }
};

template<typename T>
struct Erf {
    __device__ T operator()(T t) const { return std::erf(t); }
};

template<typename T>
struct Sigmoid {
    T one = T(1);
    __device__ T operator()(T t) const { return (one / (one + std::exp(-t))); }
};
FLOATING_HALF_BFLOAT16(log1p, Log1p);
FLOATING_HALF_BFLOAT16(erf, Erf);
FLOATING_HALF_BFLOAT16(sigmoid, Sigmoid);

FLOATING_COMPLEX_HALF(acos, Acos);
FLOATING_COMPLEX_HALF(asin, Asin);
FLOATING_COMPLEX_HALF(atan, Atan);
FLOATING_COMPLEX_HALF(cosh, Cosh);
FLOATING_COMPLEX_HALF(tan, Tan);
FLOATING_COMPLEX_HALF(sin, Sin);
FLOATING_COMPLEX_HALF(sinh, Sinh);

FLOATING_COMPLEX_HALF_BFLOAT16(exp, Exp);
FLOATING_COMPLEX_HALF_BFLOAT16(tanh, Tanh);
FLOATING_COMPLEX_HALF_BFLOAT16(log, Log);
FLOATING_COMPLEX_HALF_BFLOAT16(log10, Log10);
FLOATING_COMPLEX_HALF_BFLOAT16(log2, Log2);
FLOATING_COMPLEX_HALF_BFLOAT16(cos, Cos);
FLOATING_COMPLEX_HALF_BFLOAT16(sqrt, Sqrt);

//
// Special cases
//
std::vector<Tensor> foreach_tensor_neg_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);
    TORCH_CHECK(tensors[0].scalar_type() != kBool,
              "Negation, the `-` operator, on a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_neg_slow(tensors);
    }

    return all_types_half_bfloat16<std::negate>(tensors);
}

void foreach_tensor_neg_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);
    TORCH_CHECK(tensors[0].scalar_type() != kBool,
              "Negation, the `-` operator, on a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_neg_slow_(tensors);
    }

    all_types_half_bfloat16_<std::negate>(tensors);
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

    return floating_half<Round>(tensors);
}

void foreach_tensor_round_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_round_slow_(tensors);
    }

    floating_half_<Round>(tensors);
}

// Abs have to go via slow path in case of a complex  or integer type.
// This is because foreach kernels can't return a different dtype than passed, while 
// abs with complex or integer inputs will produce float output.
template<typename T>
struct Abs {
    __device__ T operator()(T t) const { return std::abs(t); }
};

std::vector<Tensor> foreach_tensor_abs_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);
    bool has_complex_or_integer = false;
    for (auto t : tensors) {
        if (at::isComplexType(t.scalar_type()) || 
            at::isIntegralType(t.scalar_type(), /*includeBool=*/true)) {
            has_complex_or_integer = true;
        }
    }

    if (!can_use_fast_route(tensors) || has_complex_or_integer) {
        return at::native::foreach_tensor_abs_slow(tensors);
    }

    return floating_complex_half_bfloat16<Abs>(tensors);
}

void foreach_tensor_abs_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);
    bool has_complex_or_integer = false;
    for (auto t : tensors) {
        if (at::isComplexType(t.scalar_type()) || 
            at::isIntegralType(t.scalar_type(), /*includeBool=*/true)) {
            has_complex_or_integer = true;
        }
    }

    if (!can_use_fast_route(tensors) || has_complex_or_integer) {
        return at::native::foreach_tensor_abs_slow_(tensors);
    }

    floating_complex_half_bfloat16_<Abs>(tensors);
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

    return floating_half<Trunc>(tensors);
}

void foreach_tensor_frac_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_frac_slow_(tensors);
    }

    floating_half_<Trunc>(tensors);
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

    return floating_complex_half_bfloat16<Reciprocal>(tensors);
}

void foreach_tensor_reciprocal_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_reciprocal_slow_(tensors);
    }

    floating_complex_half_bfloat16_<Reciprocal>(tensors);
}

}} // namespace at::native

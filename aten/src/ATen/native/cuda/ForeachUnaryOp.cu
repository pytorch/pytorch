#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

template <typename scalar_t, template<class> class Op> std::vector<Tensor> foreach_unary_op(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    using opmath_t = typename get_opmath_t<scalar_t>::opmath_t;
    multi_tensor_apply<2>(tensor_lists,
                          UnaryOpFunctor<scalar_t,
                                         /* depth */ 2,
                                         /* r_args_depth */ 1,
                                         /* res_arg_index */ 1>(),
                          Op<opmath_t>());

    return tensor_lists[1];
}

template <typename scalar_t, template<class> class Op> void foreach_unary_op_(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());
    using opmath_t = typename get_opmath_t<scalar_t>::opmath_t;
    multi_tensor_apply<1>(tensor_lists,
                          UnaryOpFunctor<scalar_t,
                                         /* depth */ 1,
                                         /* r_args_depth */ 1,
                                         /* res_arg_index */ 0>(),
                          Op<opmath_t>());
}

template <template<class> class Op>
std::vector<Tensor> floating_complex_half(TensorList tensors) {
    return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half,  tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void floating_complex_half_(TensorList tensors) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> all_types_complex_bfloat16_half_bool(TensorList tensors) {
    return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void all_types_complex_bfloat16_half_bool_(TensorList tensors) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> floating_complex_half_bfloat16(TensorList tensors) {
    return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void floating_complex_half_bfloat16_(TensorList tensors) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> all_types_half_complex_bfloat16(TensorList tensors) {
    return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, at::ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void all_types_half_complex_bfloat16_(TensorList tensors) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, at::ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> floating_half(TensorList tensors) {
    return AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Half,  tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void floating_half_(TensorList tensors) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> floating_half_bfloat16(TensorList tensors) {
    return AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16,  tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void floating_half_bfloat16_(TensorList tensors) {
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

// makes the functor
#define STD_FUNCTOR(op_name, functor_name)                           \
template<typename T>                                                 \
struct functor_name {                                                \
    __device__ T operator()(T t) const { return std::op_name(t); }   \
};                                                                   \

// given a functor and a "dispatch function", creates the outplace and inplace operations
#define OP_CUSTOM_FUNCTOR(function, op_name, functor_name)                \
std::vector<Tensor> foreach_tensor_##op_name##_cuda(TensorList tensors) { \
    check_foreach_api_restrictions(tensors);                              \
    if (!can_use_fast_route(tensors)) {                                   \
        return at::native::foreach_tensor_##op_name##_slow(tensors);      \
    }                                                                     \
    return function<functor_name>(tensors);                               \
}                                                                         \
void foreach_tensor_##op_name##_cuda_(TensorList tensors) {               \
    check_foreach_api_restrictions(tensors);                              \
    if (!can_use_fast_route(tensors)) {                                   \
        return at::native::foreach_tensor_##op_name##_slow_(tensors);     \
    }                                                                     \
                                                                          \
    function##_<functor_name>(tensors);                                   \
}

// creates a functor, outplace version, and inplace version.
#define OP(function, op_name, functor_name)         \
STD_FUNCTOR(op_name, functor_name);                 \
OP_CUSTOM_FUNCTOR(function, op_name, functor_name); \

OP(floating_half_bfloat16, erfc, Erfc);
OP(floating_half_bfloat16, expm1, Expm1);
OP(floating_half, lgamma, Lgamma);
OP(floating_half, trunc, Truncf);
OP(floating_half, floor, Floor);
OP(floating_half, ceil, Ceil);

OP(floating_complex_half_bfloat16, acos, Acos);
OP(floating_complex_half_bfloat16, asin, Asin);
OP(floating_complex_half_bfloat16, atan, Atan);
OP(floating_complex_half_bfloat16, cosh, Cosh);
OP(floating_complex_half_bfloat16, tan, Tan);
OP(floating_complex_half_bfloat16, sin, Sin);
OP(floating_complex_half_bfloat16, sinh, Sinh);

OP(floating_complex_half_bfloat16, exp, Exp);
OP(floating_complex_half_bfloat16, tanh, Tanh);
OP(floating_complex_half_bfloat16, log, Log);
OP(floating_complex_half_bfloat16, log10, Log10);
OP(floating_complex_half_bfloat16, log2, Log2);
OP(floating_complex_half_bfloat16, cos, Cos);
OP(floating_complex_half_bfloat16, sqrt, Sqrt);

OP(floating_half_bfloat16, log1p, Log1p);
OP(floating_half_bfloat16, erf, Erf);

//
// Special cases
// These functions must be special cased as they can't be written as std::functor_name in OP macro
//
template<typename T>
struct Sigmoid {
    T one = T(1);
    __device__ T operator()(T t) const { return (one / (one + std::exp(-t))); }
};

template<typename T>
struct Round {
    __device__ T operator()(T t) const { return std::nearbyint(t); }
};

template<typename T>
struct Trunc {
    __device__ T operator()(T t) const { return t - std::trunc(t); }
};

template<typename T>
struct Reciprocal {
    T one = T(1);
    __device__ T operator()(T t) const { return (one / t); }
};

OP_CUSTOM_FUNCTOR(floating_half_bfloat16, sigmoid, Sigmoid)
OP_CUSTOM_FUNCTOR(floating_half, round, Round)
OP_CUSTOM_FUNCTOR(floating_half, frac, Trunc)
OP_CUSTOM_FUNCTOR(floating_complex_half_bfloat16, reciprocal, Reciprocal)

std::vector<Tensor> foreach_tensor_neg_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);
    TORCH_CHECK(tensors[0].scalar_type() != kBool,
                "_foreach_neg: There is a bool tensor in the passed-in TensorList. "
                "Negation on a bool tensor is not supported. If you are trying to invert a mask, please use the `~`"
                "or `logical_not()` operator on the individual tensors instead.");

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_neg_slow(tensors);
    }

    return all_types_half_complex_bfloat16<std::negate>(tensors);
}

void foreach_tensor_neg_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);
    TORCH_CHECK(tensors[0].scalar_type() != kBool,
                "_foreach_neg: There is a bool tensor in the passed-in TensorList. "
                "Negation on a bool tensor is not supported. If you are trying to invert a mask, please use the `~`"
                "or `logical_not()` operator on the individual tensors instead.");

    if (!can_use_fast_route(tensors)) {
        return at::native::foreach_tensor_neg_slow_(tensors);
    }

    all_types_half_complex_bfloat16_<std::negate>(tensors);
}

// Abs have to go via slow path in case of a complex type.
// This is because foreach kernels can't return a different dtype than passed, while
// abs with complex inputs will produce float output.
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

    return all_types_complex_bfloat16_half_bool<Abs>(tensors);
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

    all_types_complex_bfloat16_half_bool_<Abs>(tensors);
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

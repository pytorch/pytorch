#pragma once

#include <ATen/core/DeprecatedTypeProperties.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/complex.h>
#include <c10/util/string_view.h>

#ifdef TEMPLATE_SELECTIVE_BUILD
#include <ATen/selected_mobile_ops.h>
#else
namespace at {
/**
 * The method should_include_kernel_dtype() returns true/false
 * based on whether the switching code for a specific dtype should be
 * included based on build time constants generated from tracing model
 * execution. This method will be implmeneted via code-generation and
 * included in this file when code-gen is ready.
 */
inline constexpr bool should_include_kernel_dtype(
  const char *kernel_tag_str,
  at::ScalarType scalar_type
) {
  return true;
}
}
#endif

/**
 * In the Facebook internal build (using BUCK), this macro is enabled by
 * passing in -c pt.enable_record_kernel_dtype=1 when building the tracer
 * binary.
 */
#if defined ENABLE_RECORD_KERNEL_FUNCTION_DTYPE
namespace at {
namespace detail {
TORCH_API void record_kernel_function_dtype(std::string name);
}
}

#define RECORD_KERNEL_FUNCTION_DTYPE(NAME, enum_type)                   \
  at::detail::record_kernel_function_dtype(                             \
    std::string(NAME) + "$" + toString(enum_type));
#else
#define RECORD_KERNEL_FUNCTION_DTYPE(NAME, enum_type)
#endif

#if defined __cpp_if_constexpr
#define AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...)        \
  case enum_type: {                                                              \
    if constexpr (!at::should_include_kernel_dtype(NAME, enum_type)) {           \
      AT_ERROR("dtype '", toString(enum_type), "' not selected for kernel tag ", #NAME); \
    }                                                                            \
    using HINT = type;                                                           \
    return __VA_ARGS__();                                                        \
  }
#else
#define AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...)        \
  case enum_type: {                                                              \
    at::guts::if_constexpr<(!at::should_include_kernel_dtype(NAME, enum_type))>( \
      [] {                                                                       \
        AT_ERROR("dtype '" #enum_type "' not selected for kernel tag " #NAME);   \
      }                                                                          \
    );                                                                           \
    using HINT = type;                                                           \
    return __VA_ARGS__();                                                        \
  }
#endif                                                                           \


#define AT_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...)                     \
  AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, scalar_t, __VA_ARGS__)

// Workaround for C10_UNUSED because CUDA 10.1 and below fails to handle unused
// attribute in the type aliasing context. Keep name long and verbose to avoid
// macro collisions.
#if defined(__CUDACC__) && defined(CUDA_VERSION) && CUDA_VERSION <= 10010
#define C10_UNUSED_DISPATCH_CUDA_WORKAROUND
#else
#define C10_UNUSED_DISPATCH_CUDA_WORKAROUND C10_UNUSED
#endif // defined(__CUDACC__) && defined(CUDA_VERSION) && CUDA_VERSION <= 10010

#if defined __cpp_if_constexpr
#define AT_QINT_PRIVATE_CASE_TYPE(                                           \
    NAME, enum_type, type, underlying_enum, underlying_type, ...)            \
  case enum_type: {                                                          \
    if constexpr (!at::should_include_kernel_dtype(NAME, enum_type)) {       \
      AT_ERROR("dtype '", toString(enum_type), "' not selected for kernel tag ", #NAME); \
    }                                                                        \
    using scalar_t = type;                                                   \
    using underlying_t C10_UNUSED_DISPATCH_CUDA_WORKAROUND =                 \
        scalar_t::underlying;                                                \
    const auto& SCALAR_TYPE C10_UNUSED_DISPATCH_CUDA_WORKAROUND = enum_type; \
    const auto& UNDERLYING_TYPE C10_UNUSED_DISPATCH_CUDA_WORKAROUND =        \
        toUnderlying(enum_type);                                             \
    (void)SCALAR_TYPE;  /* Suppress unused-var compiler warning */           \
    /* TODO: Use [[maybe-unused]] when C++17 becomes the standard */         \
    return __VA_ARGS__();                                                    \
  }
#else
#define AT_QINT_PRIVATE_CASE_TYPE(                                               \
    NAME, enum_type, type, underlying_enum, underlying_type, ...)                \
  case enum_type: {                                                              \
    at::guts::if_constexpr<(!at::should_include_kernel_dtype(NAME, enum_type))>( \
      [] {                                                                       \
        AT_ERROR("dtype '" #enum_type "' not selected for kernel tag " #NAME);   \
      }                                                                          \
    );                                                                           \
    using scalar_t = type;                                                       \
    using underlying_t C10_UNUSED_DISPATCH_CUDA_WORKAROUND =                     \
        scalar_t::underlying;                                                    \
    const auto& SCALAR_TYPE C10_UNUSED_DISPATCH_CUDA_WORKAROUND = enum_type;     \
    const auto& UNDERLYING_TYPE C10_UNUSED_DISPATCH_CUDA_WORKAROUND =            \
        toUnderlying(enum_type);                                                 \
    (void)SCALAR_TYPE;  /* Suppress unused-var compiler warning */               \
    /* TODO: Use [[maybe-unused]] when C++17 becomes the standard */             \
    return __VA_ARGS__();                                                        \
  }
#endif

#if defined __cpp_if_constexpr
#define AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                       \
    NAME, enum_type, type, underlying_type, bitwidth, qmin, qmax, ...)            \
  case enum_type: {                                                               \
      if constexpr (!at::should_include_kernel_dtype(NAME, enum_type)) {          \
      AT_ERROR("dtype '", toString(enum_type), "' not selected for kernel tag ", #NAME); \
    }                                                                             \
    using scalar_t = type;                                                        \
    using underlying_t C10_UNUSED_DISPATCH_CUDA_WORKAROUND =                      \
        scalar_t::underlying;                                                     \
    const auto& SCALAR_TYPE C10_UNUSED_DISPATCH_CUDA_WORKAROUND = enum_type;      \
    const auto& UNDERLYING_TYPE C10_UNUSED_DISPATCH_CUDA_WORKAROUND =             \
        toUnderlying(enum_type);                                                  \
    C10_UNUSED int bit_width = bitwidth;                                          \
    C10_UNUSED int64_t quant_min = qmin;                                          \
    C10_UNUSED int64_t quant_max = qmax;                                          \
    (void)bit_width; /* Suppress unused variable warning */                       \
    (void)quant_min; /* Suppress unused variable warning */                       \
    (void)quant_max; /* Suppress unused variable warning */                       \
    return __VA_ARGS__();                                                         \
  }
#else
#define AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                       \
    NAME, enum_type, type, underlying_type, bitwidth, qmin, qmax, ...)            \
  case enum_type: {                                                               \
      at::guts::if_constexpr<(!at::should_include_kernel_dtype(NAME, enum_type))>( \
      [] {                                                                        \
        AT_ERROR("dtype '" #enum_type "' not selected for kernel tag " #NAME);    \
      }                                                                           \
    );                                                                            \
    using scalar_t = type;                                                        \
    using underlying_t C10_UNUSED_DISPATCH_CUDA_WORKAROUND =                      \
        scalar_t::underlying;                                                     \
    const auto& SCALAR_TYPE C10_UNUSED_DISPATCH_CUDA_WORKAROUND = enum_type;      \
    const auto& UNDERLYING_TYPE C10_UNUSED_DISPATCH_CUDA_WORKAROUND =             \
        toUnderlying(enum_type);                                                  \
    int bit_width = bitwidth;                                                     \
    int64_t quant_min = qmin;                                                     \
    int64_t quant_max = qmax;                                                     \
    (void)bit_width; /* Suppress unused variable warning */                       \
    (void)quant_min; /* Suppress unused variable warning */                       \
    (void)quant_max; /* Suppress unused variable warning */                       \
    return __VA_ARGS__();                                                         \
  }
#endif

namespace detail {

inline at::ScalarType scalar_type(at::ScalarType s) {
  return s;
}

C10_DEPRECATED_MESSAGE(
    "passing at::DeprecatedTypeProperties to an AT_DISPATCH macro is deprecated, "
    "pass an at::ScalarType instead")
inline at::ScalarType scalar_type(const at::DeprecatedTypeProperties& t) {
  return t.scalarType();
}

C10_DEPRECATED_MESSAGE(
    "AT_DISPATCH_ALL_TYPES_AND_HALF is deprecated, "
    "use AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, ...) instead")
inline void deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF() {}

C10_DEPRECATED_MESSAGE(
    "AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX is deprecated, "
    "use AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, ...) "
    "instead")
inline void deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX() {}

} // namespace detail

// The AT_DISPATCH_* family of macros provides the ability to
// conveniently generate specializations of a kernel over all of the
// dtypes we care about in PyTorch.  We call it "dispatch" because
// we are "dispatching" to the correct, dtype-specific kernel.
//
// A standard usage looks like:
//
//      AT_DISPATCH_ALL_TYPES(self.scalar_type(), "op_name", [&] {
//          // Your code here, with 'scalar_t' now defined to
//          // be the dtype in question
//      })
//
// There are many variations of this macro, so it's important to
// understand exactly /which/ dtypes you want to get instantiated, as
// well as what the "default" set is.
//
// The default set of dtypes that are instantiated (e.g., by
// AT_DISPATCH_ALL_TYPES) are floating point types (float, double),
// and integral types (int32_t, int64_t, int16_t, int8_t, uint8_t),
// but NOT booleans (bool), half-precision floats (Half) or
// complex number (c10::complex<float>, c10::complex<double>).
// This "cut" is somewhat historical (the default types are the
// ones that TH historically supported), but it also reflects the
// fact that the non-default types are "poorly" behaved (booleans
// are NOT integers mod 2, half precision operations ~essentially
// don't exist on CPU, complex numbers are an experimental application).
//
// Here are the questions you should generally ask to decide which
// dispatch you want:
//
// 1. Is this an integral or floating point specific operation?
//    (If so, you'll want one of the FLOATING or INTEGRAL macros.)
//
// 2. Should half be supported?  (If you're on CPU, the answer is almost
//    definitely no.  If you do want support, use one of the AND_HALF
//    macros)
//
// Much rarer situations:
//
// 3. Should bool be supported?  (You often have to write your kernel
//    differently if arithmetic operations are involved.)  If so,
//    Use AT_DISPATCH_ALL_TYPES_AND along with ScalarType::Bool
//
// 4. Should complex be supported?  The answer is almost always no,
//    unless you are working on "generic" code that should work on
//    all dtypes.
//
// Parameters:
// -----------
//
// 1. The NAME argument is a "tag" that is used to trace and then
//    conditionally compile fragments of the case statements such
//    that the kernel functions are specialized only for the dtypes
//    that are needed. The NAME parameter *must* be a build time
//    cons char* (can't be std::string, etc...)
//
// Please ensure that the NAME is unique for every implementation
// or you run the risk of over-including code for the kernel
// functions. There is no risk of missing out on any code, so
// it's mostly a risk of a Type-2 error, and not a Type-1 error.
//

// NB: the the_type variable is not used, but we have kept it for
// backwards compatibility.  It's probably not used by anyone though;
// but we're just being safe (and it doesn't hurt.)  Note we must
// use it to shut up warnings about unused store.

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                         \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)       \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)                   \
  [&] {                                                                        \
    const auto& the_type = TYPE;                                               \
    /* don't use TYPE again in case it is an expensive or side-effect op */    \
    at::ScalarType _st = ::detail::scalar_type(the_type);                      \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                   \
    switch (_st) {                                                             \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Half, at::Half, __VA_ARGS__)  \
      default:                                                                 \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");         \
    }                                                                          \
  }()

#define AT_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)         \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(NAME,                                               \
          SCALARTYPE,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),          \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define AT_DISPATCH_FLOATING_TYPES_AND2(                                    \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                              \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...)             \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          at::ScalarType::ComplexDouble,                                    \
          c10::complex<double>,                                             \
          __VA_ARGS__)                                                      \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          at::ScalarType::ComplexFloat,                                     \
          c10::complex<float>,                                              \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(                        \
    SCALARTYPE, TYPE, NAME, ...)                                            \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),          \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                        \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                              \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                       \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                    \
    switch (_st) {                                                              \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(                                                     \
          NAME,                                                                 \
          at::ScalarType::ComplexDouble,                                        \
          c10::complex<double>,                                                 \
          __VA_ARGS__)                                                          \
      AT_PRIVATE_CASE_TYPE(                                                     \
          NAME,                                                                 \
          at::ScalarType::ComplexFloat,                                         \
          c10::complex<float>,                                                  \
          __VA_ARGS__)                                                          \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define AT_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                         \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__)     \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define AT_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)         \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(NAME,                                              \
          SCALARTYPE,                                                   \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),      \
          __VA_ARGS__)                                                  \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");  \
    }                                                                   \
  }()

#define AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                               \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                 \
    switch (_st) {                                                              \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      default:                                                                  \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");          \
    }                                                                           \
  }()

#define AT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          at::ScalarType::ComplexFloat,                                     \
          c10::complex<float>,                                              \
          __VA_ARGS__)                                                      \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          at::ScalarType::ComplexDouble,                                    \
          c10::complex<double>,                                             \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define AT_DISPATCH_QINT_TYPES(TYPE, NAME, ...)                             \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_QINT_PRIVATE_CASE_TYPE(                                            \
          NAME, at::kQInt8, at::qint8, at::kChar, int8_t, __VA_ARGS__)      \
      AT_QINT_PRIVATE_CASE_TYPE(                                            \
          NAME, at::kQUInt8, at::quint8, at::kByte, uint8_t, __VA_ARGS__)   \
      AT_QINT_PRIVATE_CASE_TYPE(                                            \
          NAME, at::kQInt32, at::qint32, at::kInt, int, __VA_ARGS__)        \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(TYPE, NAME, ...)                                   \
  [&] {                                                                                        \
    const auto& the_type = TYPE;                                                               \
    /* don't use TYPE again in case it is an expensive or side-effect op */                    \
    at::ScalarType _st = ::detail::scalar_type(the_type);                                      \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                                   \
    switch (_st) {                                                                             \
      AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                                      \
          NAME, at::kQInt8, at::qint8, int8_t, CHAR_BIT, SCHAR_MIN, SCHAR_MAX, __VA_ARGS__)    \
      AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                                      \
          NAME, at::kQUInt8, at::quint8, uint8_t, CHAR_BIT, 0, UCHAR_MAX, __VA_ARGS__)         \
      AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                                      \
          NAME, at::kQInt32, at::qint32, int, CHAR_BIT * sizeof(int), INT_MIN, INT_MAX, __VA_ARGS__) \
      AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                                      \
          NAME, at::kQUInt4x2, at::quint4x2, uint8_t, 4, 0, 15, __VA_ARGS__)                   \
      AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                                      \
          NAME, at::kQUInt2x4, at::quint2x4, uint8_t, 2, 0, 3, __VA_ARGS__)                   \
      default:                                                                                 \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                        \
    }                                                                                          \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...)                  \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op*/  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME,                                                  \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME,                                                  \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__) \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define AT_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)          \
  [&] {                                                                 \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op*/  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(                                             \
          NAME,                                                         \
          SCALARTYPE,                                                   \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),      \
          __VA_ARGS__)                                                  \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");  \
    }                                                                   \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, ...)  \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op*/  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                         \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          at::ScalarType::ComplexFloat,                                     \
          c10::complex<float>,                                              \
          __VA_ARGS__)                                                      \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          at::ScalarType::ComplexDouble,                                    \
          c10::complex<double>,                                             \
          __VA_ARGS__)                                                      \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),          \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define AT_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  [&] {                                                                       \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op*/  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)         \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)         \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)         \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(                                                   \
          NAME,                                                               \
          SCALARTYPE1,                                                        \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),           \
          __VA_ARGS__)                                                        \
      AT_PRIVATE_CASE_TYPE(                                                   \
          NAME,                                                               \
          SCALARTYPE2,                                                        \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),           \
          __VA_ARGS__)                                                        \
      default:                                                                \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");        \
    }                                                                         \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(                             \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                              \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op*/  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(                                                       \
          NAME, at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(                                                       \
          NAME, at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define AT_DISPATCH_ALL_TYPES_AND3(                                         \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)                 \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op*/  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(                                             \
          NAME,                                                         \
          SCALARTYPE1,                                                  \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),     \
          __VA_ARGS__)                                                  \
      AT_PRIVATE_CASE_TYPE(                                             \
          NAME,                                                         \
          SCALARTYPE2,                                                  \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),     \
          __VA_ARGS__)                                                  \
      AT_PRIVATE_CASE_TYPE(                                             \
          NAME,                                                         \
          SCALARTYPE3,                                                  \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t),     \
          __VA_ARGS__)                                                  \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");  \
    }                                                                   \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(                             \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)                 \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op*/  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(                                                       \
          NAME, at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(                                                       \
          NAME, at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME,                                                             \
          SCALARTYPE3,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define AT_DISPATCH_INDEX_TYPES(TYPE, NAME, ...)                            \
  [&] {                                                                     \
    const auto& the_index_type = TYPE;                                      \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _it = ::detail::scalar_type(the_index_type);             \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _it)                                 \
    switch (_it) {                                                          \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Int, int32_t, index_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Long, int64_t, index_t, __VA_ARGS__)\
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_it), "'");      \
    }                                                                       \
  }()

// ----------------------------------------------------------------------------
// DEPRECATED MACROS, DON'T USE THESE
// ----------------------------------------------------------------------------

#define AT_DISPATCH_ALL_TYPES_AND_HALF(TYPE, NAME, ...)                     \
  [&] {                                                                     \
    detail::deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF();                    \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Half, at::Half, __VA_ARGS__)     \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

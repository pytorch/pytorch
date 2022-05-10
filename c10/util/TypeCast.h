#pragma once
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cstring>
#include <type_traits>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace c10 {

namespace detail {

template <bool /*false*/>
struct maybe_real {
  template <typename src_t>
  C10_HOST_DEVICE static inline constexpr src_t apply(src_t src) {
    return src;
  }
};

template <>
struct maybe_real<true> {
  template <typename src_t>
  C10_HOST_DEVICE static inline constexpr decltype(auto) apply(src_t src) {
    return src.real();
  }
};

template <typename dest_t, typename src_t>
C10_HOST_DEVICE constexpr auto convert_preprocess(src_t src) {
  return src;
}

template <typename dest_t, typename src_real_t>
C10_HOST_DEVICE constexpr auto convert_preprocess(
    c10::complex<src_real_t> src) {
  return maybe_real<!c10::is_complex<dest_t>::value>::apply(src);
}

template <typename dest_t>
C10_HOST_DEVICE constexpr int convert_preprocess(bool src) {
  // Normalize boolean values to 1 or 0 (gh-54789)
  // memcpy is needed to work-around optimizations in the compiler,
  // otherise it will just generate a mov instruction
  char tmp = 0;
  std::memcpy(&tmp, &src, 1);
  return tmp ? 1 : 0;
}

// Note: deliberately ignores undefined behavior, consistent with NumPy.
// PyTorch's type conversions can cause a variety of undefined behavior,
// including float to integral overflow and signed to unsigned integer overflow.
// Some of this undefined behavior is addressed below.
template <typename dest_t, typename src_t>
struct static_cast_with_inter_type {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline dest_t apply(
      src_t src) {
    return static_cast<dest_t>(src);
  }
};

// Partial template instantiation for casting to uint8.
// Note: Converting from negative float values to unsigned integer types is
// undefined behavior in C++, and current CPU and GPU compilers exhibit
// divergent behavior. Casting from negative float values to signed
// integer types and then to unsigned integer types is not undefined,
// however, so this cast improves the consistency of type conversions
// to uint8 across compilers.
// Further note: Type conversions across compilers still have other undefined
// and divergent behavior.
template <typename src_t>
struct static_cast_with_inter_type<uint8_t, src_t> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline uint8_t apply(
      src_t src) {
    return static_cast<uint8_t>(static_cast<int64_t>(src));
  }
};

template <>
struct static_cast_with_inter_type<c10::complex<c10::Half>, c10::BFloat16> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::BFloat16 src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<c10::complex<c10::Half>, c10::Half> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::Half src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::complex<double>> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::complex<double> src) {
    return static_cast<c10::complex<c10::Half>>(
        static_cast<c10::complex<float>>(src));
  }
};

} // namespace detail

template <typename To, typename From>
constexpr C10_HOST_DEVICE C10_ALWAYS_INLINE To convert(From f) {
  auto tmp = c10::detail::convert_preprocess<To>(f);
  return c10::detail::static_cast_with_inter_type<To, decltype(tmp)>::apply(
      tmp);
}

// Define separately to avoid being inlined and prevent code-size bloat
C10_API void report_overflow(const char* name);

template <typename To, typename From>
To checked_convert(From f, const char* name) {
  // Converting to bool can't overflow so we exclude this case from checking.
  if (!std::is_same<To, bool>::value && overflows<To, From>(f)) {
    report_overflow(name);
  }
  return convert<To, From>(f);
}

// Dynamic type casting utils:
// - fetch_and_cast
// - cast_and_store
//
// fetch_and_cast fetch a value with dynamic type specified by a ScalarType
// from a void pointer and cast it to a static type.
//
// cast_and_store casts a static typed value into dynamic type specified
// by a ScalarType, and store it into a void pointer.
//
// NOTE:
//
// Dynamic casting allows us to support type promotion without blowing up
// the combination space: For example, without dynamic cast, in order to
// implement `add_` with type promotion, we would need something like
//
// AT_DISPATCH_ALL_TYPES(output.dtype(),
//    AT_DISPATCH_ALL_TYPES(input1.dtype(),
//       AT_DISPATCH_ALL_TYPES(input2.dtype(),
//           [](arg0_t a, arg1_t b) -> out_t { return a + b; }
//       )
//    )
// )
//
// If we support N dtypes, the above code would generate the a+b kernel for
// all the N * N * N different supported types, the compilation time and
// binary size would become horrible.
//
// Dynamic casting might sounds like a bad idea in terms of performance.
// Especially if you ever do it in a loop, you are going to do a billion tests.
// But in practice it is not as bad as it might look:
//
// - on CPU, this is a branch that always has the same outcome, therefore
//   hopefully the branch predictor could do the job pretty well
// - on GPU, these branches will not diverge, so we could still have the same
//   warp executing the same line of code
// - Most kernels, like `add`, are bandwidth bound, adding a few clock cycles to
//   check an integer does not hurt the performance much because the ALUs would
//   wait for load instructions anyway.
//
// For the discussion and benchmark, refer to:
// - https://github.com/pytorch/pytorch/pull/28343
// - https://github.com/pytorch/pytorch/pull/28344
// - https://github.com/pytorch/pytorch/pull/28345
//

#ifdef C10_HOST_DEVICE
#define ERROR_UNSUPPORTED_CAST CUDA_KERNEL_ASSERT(false);
#else
#define ERROR_UNSUPPORTED_CAST TORCH_CHECK(false, "Unexpected scalar type");
#endif

// Fetch a value with dynamic type src_type from ptr, and cast it to static type
// dest_t.
#define FETCH_AND_CAST_CASE(type, scalartype) \
  case ScalarType::scalartype:                \
    return c10::convert<dest_t>(*(const type*)ptr);

template <typename dest_t>
C10_HOST_DEVICE inline dest_t fetch_and_cast(
    const ScalarType src_type,
    const void* ptr) {
  switch (src_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(FETCH_AND_CAST_CASE)
    default:
      ERROR_UNSUPPORTED_CAST
  }
  return dest_t(0); // just to avoid compiler warning
}

// Cast a value with static type src_t into dynamic dest_type, and store it to
// ptr.
#define CAST_AND_STORE_CASE(type, scalartype) \
  case ScalarType::scalartype:                \
    *(type*)ptr = c10::convert<type>(value);  \
    return;
template <typename src_t>
C10_HOST_DEVICE inline void cast_and_store(
    const ScalarType dest_type,
    void* ptr,
    src_t value) {
  switch (dest_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(CAST_AND_STORE_CASE)
    default:;
  }
  ERROR_UNSUPPORTED_CAST
}

#define DEFINE_UNCASTABLE(T, scalartype_)                     \
  template <>                                                 \
  C10_HOST_DEVICE inline T fetch_and_cast<T>(                 \
      const ScalarType src_type, const void* ptr) {           \
    CUDA_KERNEL_ASSERT(ScalarType::scalartype_ == src_type);  \
    return *(const T*)ptr;                                    \
  }                                                           \
  template <>                                                 \
  C10_HOST_DEVICE inline void cast_and_store<T>(              \
      const ScalarType dest_type, void* ptr, T value) {       \
    CUDA_KERNEL_ASSERT(ScalarType::scalartype_ == dest_type); \
    *(T*)ptr = value;                                         \
  }

AT_FORALL_QINT_TYPES(DEFINE_UNCASTABLE)

#undef FETCH_AND_CAST_CASE
#undef CAST_AND_STORE_CASE
#undef DEFINE_UNCASTABLE
#undef ERROR_UNSUPPORTED_CAST

} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()

// Trigger tests for D25440771. TODO: Remove this line any time you want.

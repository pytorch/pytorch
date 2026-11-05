#pragma once

#include <ATen/native/RoundedConversion.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#if !defined(USE_ROCM)
#include <cuda.h>
#include <cuda_fp8.h>
#endif

namespace at::native {

// Each conversion below uses a hardware instruction where one exists for the
// (src, dst, mode) combination and the target architecture, and otherwise the
// portable software conversion, which produces the same values (NaN may be
// encoded differently).
template <typename dst_t>
C10_HOST_DEVICE inline dst_t
convert_rounded_from_float(float x, c10::RoundingMode mode) {
#if defined(__CUDA_ARCH__) && !defined(USE_ROCM)
  using c10::RoundingMode;
  if constexpr (std::is_same_v<dst_t, Half>) {
    switch (mode) {
      case RoundingMode::ToNearestEven:
        return __float2half_rn(x);
      case RoundingMode::TowardZero:
        return __float2half_rz(x);
      case RoundingMode::TowardPositive:
        return __float2half_ru(x);
      case RoundingMode::TowardNegative:
        return __float2half_rd(x);
    }
  } else if constexpr (std::is_same_v<dst_t, BFloat16>) {
    // These use cvt.rn/rz.bf16.f32 on sm_80+ and cvt.rp/rm.bf16.f32 on
    // sm_90+, and emulate them otherwise.
    switch (mode) {
      case RoundingMode::ToNearestEven:
        return __float2bfloat16_rn(x);
      case RoundingMode::TowardZero:
        return __float2bfloat16_rz(x);
      case RoundingMode::TowardPositive:
        return __float2bfloat16_ru(x);
      case RoundingMode::TowardNegative:
        return __float2bfloat16_rd(x);
    }
#if __CUDA_ARCH__ >= 890
  } else if constexpr (
      std::is_same_v<dst_t, Float8_e4m3fn> ||
      std::is_same_v<dst_t, Float8_e5m2>) {
    using L = std::numeric_limits<dst_t>;
    using nv_t = std::conditional_t<
        std::is_same_v<dst_t, Float8_e4m3fn>,
        __nv_fp8_e4m3,
        __nv_fp8_e5m2>;
    constexpr bool kSaturates = c10::detail::Saturates<dst_t>::value;
    const float kDstMaxFinite = L::max();
    // The hardware only rounds to nearest, saturating. The other modes step
    // the result by one encoding toward x when it is on the wrong side of x.
    nv_t y(x);
    if (mode != RoundingMode::ToNearestEven) {
      const bool neg = signbit(x);
      const bool to_zero = mode == RoundingMode::TowardZero ||
          (mode == RoundingMode::TowardPositive && neg) ||
          (mode == RoundingMode::TowardNegative && !neg);
      const float rounded = fabsf(static_cast<float>(y));
      // Comparisons with NaN are false, so NaN is left alone. Consecutive
      // magnitudes have consecutive encodings, so without saturation stepping
      // past the largest finite value gives infinity.
      const bool step_down = to_zero && rounded > fabsf(x);
      const bool step_up = !to_zero && rounded < fabsf(x) &&
          !(kSaturates && rounded == kDstMaxFinite);
      y.__x += static_cast<int>(step_up) - static_cast<int>(step_down);
    }
    if constexpr (!kSaturates) {
      // Where the hardware saturates, dst_t has infinities: for infinite x,
      // and when rounding to nearest from halfway between the largest finite
      // value and the next power of two.
      const float midpoint =
          0.5f * (kDstMaxFinite + static_cast<float>(1 << L::max_exponent));
      if (isinf(x) ||
          (mode == RoundingMode::ToNearestEven && fabsf(x) >= midpoint)) {
        return signbit(x) ? -L::infinity() : L::infinity();
      }
    }
    return dst_t(y.__x, dst_t::from_bits());
#endif
#if CUDA_VERSION >= 12080
  } else if constexpr (std::is_same_v<dst_t, Float8_e8m0fnu>) {
    // Uses cvt.rz/rp.ue8m0x2.f32 when built for sm_100a and similar targets
    // (RoundedConversionArchSpecific.cu, see cmake/Codegen.cmake), and
    // emulates it otherwise.
    // The unsigned format rounds the magnitude, so toward_negative is
    // toward_zero. There is no round-to-nearest instruction.
    if (mode != RoundingMode::ToNearestEven) {
      const auto bits = __nv_cvt_float_to_e8m0(
          x,
          __NV_NOSAT,
          mode == RoundingMode::TowardPositive ? cudaRoundPosInf
                                               : cudaRoundZero);
      return Float8_e8m0fnu(bits, Float8_e8m0fnu::from_bits());
    }
#endif
#if CUDA_VERSION >= 13040 && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
    __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1070 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1100
  } else if constexpr (std::is_same_v<dst_t, Float8_e5m3fnu>) {
    // cvt.rn/rz/rp.satfinite.ue5m3x2.f32, which only sm_107f and similar
    // targets have (elsewhere the intrinsic is emulated, and our conversion is
    // used instead). The unsigned format rounds the magnitude, so
    // toward_negative is toward_zero, and it saturates like the default
    // conversion.
    const auto rounding = mode == RoundingMode::ToNearestEven
        ? cudaRoundNearest
        : mode == RoundingMode::TowardPositive ? cudaRoundPosInf
                                               : cudaRoundZero;
    return Float8_e5m3fnu(
        __nv_cvt_float_to_ue5m3(x, __NV_SATFINITE, rounding),
        Float8_e5m3fnu::from_bits());
#endif
  }
#endif
  return c10::convert_with_rounding<dst_t>(x, mode);
}

template <typename dst_t>
C10_HOST_DEVICE inline dst_t
convert_rounded_from_double(double x, c10::RoundingMode mode) {
#if defined(__CUDA_ARCH__) && !defined(USE_ROCM)
  using c10::RoundingMode;
  if constexpr (std::is_same_v<dst_t, float>) {
    switch (mode) {
      case RoundingMode::ToNearestEven:
        return __double2float_rn(x);
      case RoundingMode::TowardZero:
        return __double2float_rz(x);
      case RoundingMode::TowardPositive:
        return __double2float_ru(x);
      case RoundingMode::TowardNegative:
        return __double2float_rd(x);
    }
  } else if constexpr (std::is_same_v<dst_t, Half>) {
    unsigned short bits;
    switch (mode) {
      case RoundingMode::ToNearestEven:
        asm("cvt.rn.f16.f64 %0, %1;" : "=h"(bits) : "d"(x));
        break;
      case RoundingMode::TowardZero:
        asm("cvt.rz.f16.f64 %0, %1;" : "=h"(bits) : "d"(x));
        break;
      case RoundingMode::TowardPositive:
        asm("cvt.rp.f16.f64 %0, %1;" : "=h"(bits) : "d"(x));
        break;
      case RoundingMode::TowardNegative:
        asm("cvt.rm.f16.f64 %0, %1;" : "=h"(bits) : "d"(x));
        break;
    }
    return Half(bits, Half::from_bits());
#if __CUDA_ARCH__ >= 900
  } else if constexpr (std::is_same_v<dst_t, BFloat16>) {
    unsigned short bits;
    switch (mode) {
      case RoundingMode::ToNearestEven:
        asm("cvt.rn.bf16.f64 %0, %1;" : "=h"(bits) : "d"(x));
        break;
      case RoundingMode::TowardZero:
        asm("cvt.rz.bf16.f64 %0, %1;" : "=h"(bits) : "d"(x));
        break;
      case RoundingMode::TowardPositive:
        asm("cvt.rp.bf16.f64 %0, %1;" : "=h"(bits) : "d"(x));
        break;
      case RoundingMode::TowardNegative:
        asm("cvt.rm.bf16.f64 %0, %1;" : "=h"(bits) : "d"(x));
        break;
    }
    return BFloat16(bits, BFloat16::from_bits());
#endif
  } else {
    return convert_rounded_from_float<dst_t>(
        c10::detail::round_to_odd_float(x), mode);
  }
#endif
  return c10::convert_with_rounding<dst_t>(x, mode);
}

// Pairs rejected by check_rounded_conversion are not instantiated, which needs
// a template: AT_DISPATCH's lambdas are not.
template <typename src_t, typename dst_t>
void to_dtype_rounded_loop(TensorIteratorBase& iter, c10::RoundingMode mode) {
  if constexpr (
      !std::is_same_v<src_t, dst_t> &&
      (std::is_same_v<src_t, double> || !std::is_same_v<dst_t, float>)) {
    gpu_kernel_nocast(iter, [mode] GPU_LAMBDA(src_t x) -> dst_t {
      if constexpr (std::is_same_v<src_t, double>) {
        return convert_rounded_from_double<dst_t>(x, mode);
      } else {
        // Half and BFloat16 widen to float exactly.
        return convert_rounded_from_float<dst_t>(static_cast<float>(x), mode);
      }
    });
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported rounded conversion");
  }
}

// The float8_e8m0fnu and float8_e5m3fnu destinations, whose instructions need
// arch-specific targets, are in a separate file that is also built for them.
void to_dtype_rounded_kernel_cuda_arch_specific(
    TensorIteratorBase& iter,
    c10::RoundingMode mode);

} // namespace at::native

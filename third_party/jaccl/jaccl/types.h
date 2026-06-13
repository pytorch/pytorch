// Copyright © 2025 Apple Inc.

#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>

#if defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace jaccl {

namespace {

union float_bits {
  float f;
  uint32_t u;
};

} // namespace

#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC

#include <arm_fp16.h>

#else

#define __JACCL_HALF_NAN__ 0x7D00

//
// The MLX float16 compatibility fallback. Redefined here to keep JACCL
// standalone. Also it doesn't really need all the ops MXL defines.
//
struct float16_t {
  uint16_t bits_;

  float16_t(const float& x) : bits_(0) {
    float_bits in;
    in.f = x;

    // Extract sign
    uint32_t x_sign_32 = in.u & uint32_t(0x80000000);
    uint16_t x_sign_16 = (x_sign_32 >> 16);

    if (std::isnan(x)) {
      bits_ = x_sign_16 | uint16_t(__JACCL_HALF_NAN__);
    } else {
      // Union
      float_bits inf_scale, zero_scale, magic_bits;

      // Find exponent bits and take the max supported by half
      uint32_t x_expo_32 = in.u & uint32_t(0x7f800000);
      uint32_t max_expo_32 = uint32_t(0x38800000);
      x_expo_32 = x_expo_32 < max_expo_32 ? max_expo_32 : x_expo_32;
      x_expo_32 += uint32_t(15) << 23;

      // Handle scaling to inf as needed
      inf_scale.u = uint32_t(0x77800000);
      zero_scale.u = uint32_t(0x08800000);

      // Combine with magic and let addition do rounding
      magic_bits.u = x_expo_32;
      magic_bits.f += (std::abs(x) * inf_scale.f) * zero_scale.f;

      // Take the lower 5 bits of the exponent
      uint32_t x_expo_16 = ((magic_bits.u >> 13) & uint32_t(0x7c00));

      // Collect the lower 12 bits which have the mantissa
      uint32_t x_mant_16 = magic_bits.u & uint32_t(0x0fff);

      // Combine sign, exp and mantissa
      bits_ = (x_sign_16 | uint16_t(x_expo_16 + x_mant_16));
    }
  }

  operator float() const {
    float_bits out;

    uint32_t x_sign_32 = (bits_ << 16) & uint32_t(0x80000000);
    uint32_t base = (bits_ << 16);
    uint32_t two_base = base + base;

    uint32_t denorm_max = 1u << 27;
    if (two_base < denorm_max) {
      out.u = uint32_t(126) << 23; // magic mask
      out.u |= (two_base >> 17); // Bits from fp16
      out.f -= 0.5f; // magic bias
    } else {
      out.u = uint32_t(0xE0) << 23; // exponent offset
      out.u += (two_base >> 4); // Bits from fp16
      float out_unscaled = out.f; // Store value
      out.u = uint32_t(0x7800000); // exponent scale
      out.f *= out_unscaled;
    }

    // Add sign
    out.u |= x_sign_32;

    return out.f;
  }

  bool operator<(float16_t x) {
    return static_cast<float>(*this) < static_cast<float>(x);
  }

  bool operator>(float16_t x) {
    return static_cast<float>(*this) > static_cast<float>(x);
  }

  float16_t operator+(float16_t x) {
    return static_cast<float>(*this) + static_cast<float>(x);
  }

  float16_t& operator+=(float16_t x) {
    *this = *this + x;
    return *this;
  }
};

#endif

//
// Check at runtime if the CPU supports native bfloat16 (FEAT_BF16).
//
// This allows us to compile once for all Macs but still enable the feature if
// it is supported.
//
inline bool has_native_bf16_support() {
#if defined(__aarch64__) && defined(__APPLE__)
  static bool has_support = []() {
    int value = 0;
    size_t value_size = sizeof(value);
    int success = sysctlbyname(
        "hw.optional.arm.FEAT_BF16", &value, &value_size, nullptr, 0);
    return success == 0 & value != 0;
  }();
  return has_support;
#else
  return false;
#endif
}

//
// The MLX bfloat16 compatibility fallback.
//
// Contrary to float16 we always define it and we 'll use
// has_native_bf16_support to decide if we are going to use __bf16 instead
// during runtime.
//

#define __JACCL_BFLOAT_NAN__ 0x7FC0

struct bfloat16_t {
  uint16_t bits_;

  bfloat16_t(const float& x) {
    if (std::isnan(x)) {
      bits_ = __JACCL_BFLOAT_NAN__;
    } else {
      float_bits in;
      in.f = x;
      in.u += (in.u >> 16 & 1) + uint32_t(0x7FFF);
      bits_ = in.u >> 16;
    }
  }

  operator float() const {
    float_bits out;
    out.u = ((uint32_t)bits_) << 16;
    return out.f;
  }

  bool operator<(bfloat16_t x) {
    return static_cast<float>(*this) < static_cast<float>(x);
  }

  bool operator>(bfloat16_t x) {
    return static_cast<float>(*this) > static_cast<float>(x);
  }

  bfloat16_t operator+(bfloat16_t x) {
    return static_cast<float>(*this) + static_cast<float>(x);
  }

  bfloat16_t& operator+=(bfloat16_t x) {
    *this = *this + x;
    return *this;
  }
};

using complex64_t = std::complex<float>;

inline bool operator<(complex64_t lhs, complex64_t rhs) {
  return lhs.real() < rhs.real() ||
      (lhs.real() == rhs.real() && lhs.imag() < rhs.imag());
}

inline bool operator>(complex64_t lhs, complex64_t rhs) {
  return lhs.real() > rhs.real() ||
      (lhs.real() == rhs.real() && lhs.imag() > rhs.imag());
}

template <typename T>
struct type_identity {
  using type = T;
};

#define JACCL_GET_TYPE(x) typename decltype(x)::type

/**
 * Dispatch a function for all supported types based on a Dtype.
 */
template <typename F>
void dispatch_all_types(int dtype, F&& f) {
  switch (dtype) {
    case Dtype::Bool:
      f(type_identity<bool>{});
      break;
    case Dtype::Int8:
      f(type_identity<int8_t>{});
      break;
    case Dtype::Int16:
      f(type_identity<int16_t>{});
      break;
    case Dtype::Int32:
      f(type_identity<int32_t>{});
      break;
    case Dtype::Int64:
      f(type_identity<int64_t>{});
      break;
    case Dtype::UInt8:
      f(type_identity<uint8_t>{});
      break;
    case Dtype::UInt16:
      f(type_identity<uint16_t>{});
      break;
    case Dtype::UInt32:
      f(type_identity<uint32_t>{});
      break;
    case Dtype::UInt64:
      f(type_identity<uint64_t>{});
      break;
    case Dtype::Float16:
      f(type_identity<float16_t>{});
      break;
    case Dtype::BFloat16:
      f(type_identity<bfloat16_t>{});
      break;
    case Dtype::Float32:
      f(type_identity<float>{});
      break;
    case Dtype::Float64:
      f(type_identity<double>{});
      break;
    case Dtype::Complex64:
      f(type_identity<complex64_t>{});
      break;
  }
}

} // namespace jaccl

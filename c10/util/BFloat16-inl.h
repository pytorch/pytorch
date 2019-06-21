#pragma once

#include <c10/macros/Macros.h>
#include <limits>

namespace c10 {

/// Constructors

inline C10_HOST_DEVICE BFloat16::BFloat16(float value) {
  uint32_t res;
#if defined(__CUDA_ARCH__)
  cudaMemcpy(&res, &value, sizeof(res), cudaMemcpyDeviceToHost);
#else
  std::memcpy(&res, &value, sizeof(res));
#endif
  val_ = res >> 16;
}

/// Implicit conversions
inline C10_HOST_DEVICE BFloat16::operator float() const {
  float res = 0;
  uint32_t tmp = val_;
  tmp <<= 16;
  std::memcpy(&res, &tmp, sizeof(tmp));
  return res;
}

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::BFloat16> {
  public:
    static constexpr bool is_signed = true;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr c10::BFloat16 lowest() {
      return at::BFloat16(0xFBFF, at::BFloat16::from_bits());
    }
    static constexpr c10::BFloat16 max() {
      return at::BFloat16(0x7BFF, at::BFloat16::from_bits());
    }
};

} // namespace std

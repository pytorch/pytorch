#pragma once

#include <complex>
#include <cstdint>
#include <functional>

#include "lazy_tensors/core/lib/bfloat16/bfloat16.h"

namespace lazy_tensors {

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

struct half {
  half() { LTC_LOG(FATAL) << "Not implemented yet."; }

  explicit half(const float v) { LTC_LOG(FATAL) << "Not implemented yet."; }

  explicit operator float() const { LTC_LOG(FATAL) << "Not implemented yet."; }
};

}  // namespace lazy_tensors

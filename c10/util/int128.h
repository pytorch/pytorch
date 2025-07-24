// This file is based on the uint128 implementation of protobuf at
// https://github.com/protocolbuffers/protobuf/blob/1e88936fce10cf773cb72b44c6a7f48b38c7578b/src/google/protobuf/stubs/int128.h
//
// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <c10/macros/Export.h>
#include <cstdint>
#include <iosfwd>

namespace c10 {

struct uint128_pod;

// TODO(xiaofeng): Define GOOGLE_PROTOBUF_HAS_CONSTEXPR when constexpr is
// available.
#ifdef GOOGLE_PROTOBUF_HAS_CONSTEXPR
#define UINT128_CONSTEXPR constexpr
#else
#define UINT128_CONSTEXPR
#endif

class uint128;
inline uint128& operator<<=(uint128& self, int amount);

// An unsigned 128-bit integer type. Thread-compatible.
class C10_API uint128 {
 public:
  UINT128_CONSTEXPR uint128(); // Sets to 0, but don't trust on this behavior.
  UINT128_CONSTEXPR uint128(uint64_t top, uint64_t bottom);
#ifndef SWIG
  UINT128_CONSTEXPR uint128(int bottom);
  UINT128_CONSTEXPR uint128(uint32_t bottom); // Top 96 bits = 0
#endif
  UINT128_CONSTEXPR uint128(uint64_t bottom); // hi_ = 0
  UINT128_CONSTEXPR uint128(const uint128_pod& val);

  // Trivial copy constructor, assignment operator and destructor.

  void Initialize(uint64_t top, uint64_t bottom);

  // Arithmetic operators.
  uint128& operator+=(const uint128& b);
  uint128& operator-=(const uint128& b);
  uint128& operator*=(const uint128& b);
  // Long division/modulo for uint128.
  uint128& operator/=(const uint128& b);
  uint128& operator%=(const uint128& b);
  uint128 operator++(int);
  uint128 operator--(int);
  // Make msvc happy with using operator<<= from DivModImpl
  // which is a static function, and linker complained about missing
  // static version of this overload
  friend uint128& operator<<=(uint128&, int);
  uint128& operator>>=(int);
  uint128& operator&=(const uint128& b);
  uint128& operator|=(const uint128& b);
  uint128& operator^=(const uint128& b);
  uint128& operator++();
  uint128& operator--();

  friend uint64_t Uint128Low64(const uint128& v);
  friend uint64_t Uint128High64(const uint128& v);

  // We add "std::" to avoid including all of port.h.
  C10_API friend std::ostream& operator<<(std::ostream& o, const uint128& b);

 private:
  static void DivModImpl(
      uint128 dividend,
      uint128 divisor,
      uint128* quotient_ret,
      uint128* remainder_ret);

  // Little-endian memory order optimizations can benefit from
  // having lo_ first, hi_ last.
  // See util/endian/endian.h and Load128/Store128 for storing a uint128.
  uint64_t lo_;
  uint64_t hi_;

  // Not implemented, just declared for catching automatic type conversions.
  uint128(uint8_t);
  uint128(uint16_t);
  uint128(float v);
  uint128(double v);
};

// This is a POD form of uint128 which can be used for static variables which
// need to be operated on as uint128.
struct uint128_pod {
  // Note: The ordering of fields is different than 'class uint128' but the
  // same as its 2-arg constructor.  This enables more obvious initialization
  // of static instances, which is the primary reason for this struct in the
  // first place.  This does not seem to defeat any optimizations wrt
  // operations involving this struct.
  uint64_t hi;
  uint64_t lo;
};

C10_API extern const uint128_pod kuint128max;

// allow uint128 to be logged
C10_API extern std::ostream& operator<<(std::ostream& o, const uint128& b);

// Methods to access low and high pieces of 128-bit value.
// Defined externally from uint128 to facilitate conversion
// to native 128-bit types when compilers support them.
inline uint64_t Uint128Low64(const uint128& v) {
  return v.lo_;
}
inline uint64_t Uint128High64(const uint128& v) {
  return v.hi_;
}

// TODO: perhaps it would be nice to have int128, a signed 128-bit type?

// --------------------------------------------------------------------------
//                      Implementation details follow
// --------------------------------------------------------------------------
inline bool operator==(const uint128& lhs, const uint128& rhs) {
  return (
      Uint128Low64(lhs) == Uint128Low64(rhs) &&
      Uint128High64(lhs) == Uint128High64(rhs));
}
inline bool operator!=(const uint128& lhs, const uint128& rhs) {
  return !(lhs == rhs);
}

inline UINT128_CONSTEXPR uint128::uint128() : lo_(0), hi_(0) {}
inline UINT128_CONSTEXPR uint128::uint128(uint64_t top, uint64_t bottom)
    : lo_(bottom), hi_(top) {}
inline UINT128_CONSTEXPR uint128::uint128(const uint128_pod& v)
    : lo_(v.lo), hi_(v.hi) {}
inline UINT128_CONSTEXPR uint128::uint128(uint64_t bottom)
    : lo_(bottom), hi_(0) {}
#ifndef SWIG
inline UINT128_CONSTEXPR uint128::uint128(uint32_t bottom)
    : lo_(bottom), hi_(0) {}
inline UINT128_CONSTEXPR uint128::uint128(int bottom)
    : lo_(bottom), hi_(static_cast<int64_t>((bottom < 0) ? -1 : 0)) {}
#endif

#undef UINT128_CONSTEXPR

inline void uint128::Initialize(uint64_t top, uint64_t bottom) {
  hi_ = top;
  lo_ = bottom;
}

// Comparison operators.

#define CMP128(op)                                                  \
  inline bool operator op(const uint128& lhs, const uint128& rhs) { \
    return (Uint128High64(lhs) == Uint128High64(rhs))               \
        ? (Uint128Low64(lhs) op Uint128Low64(rhs))                  \
        : (Uint128High64(lhs) op Uint128High64(rhs));               \
  }

CMP128(<)
CMP128(>)
CMP128(>=)
CMP128(<=)

#undef CMP128

// Unary operators

inline uint128 operator-(const uint128& val) {
  const uint64_t hi_flip = ~Uint128High64(val);
  const uint64_t lo_flip = ~Uint128Low64(val);
  const uint64_t lo_add = lo_flip + 1;
  if (lo_add < lo_flip) {
    return uint128(hi_flip + 1, lo_add);
  }
  return uint128(hi_flip, lo_add);
}

inline bool operator!(const uint128& val) {
  return !Uint128High64(val) && !Uint128Low64(val);
}

// Logical operators.

inline uint128 operator~(const uint128& val) {
  return uint128(~Uint128High64(val), ~Uint128Low64(val));
}

#define LOGIC128(op)                                                   \
  inline uint128 operator op(const uint128& lhs, const uint128& rhs) { \
    return uint128(                                                    \
        Uint128High64(lhs) op Uint128High64(rhs),                      \
        Uint128Low64(lhs) op Uint128Low64(rhs));                       \
  }

LOGIC128(|)
LOGIC128(&)
LOGIC128(^)

#undef LOGIC128

#define LOGICASSIGN128(op)                                      \
  inline uint128& uint128::operator op(const uint128 & other) { \
    hi_ op other.hi_;                                           \
    lo_ op other.lo_;                                           \
    return *this;                                               \
  }

LOGICASSIGN128(|=)
LOGICASSIGN128(&=)
LOGICASSIGN128(^=)

#undef LOGICASSIGN128

// Shift operators.

inline uint128 operator<<(const uint128& val, int amount) {
  // uint64_t shifts of >= 64 are undefined, so we will need some
  // special-casing.
  if (amount < 64) {
    if (amount == 0) {
      return val;
    }
    uint64_t new_hi =
        (Uint128High64(val) << amount) | (Uint128Low64(val) >> (64 - amount));
    uint64_t new_lo = Uint128Low64(val) << amount;
    return uint128(new_hi, new_lo);
  } else if (amount < 128) {
    return uint128(Uint128Low64(val) << (amount - 64), 0);
  } else {
    return uint128(0, 0);
  }
}

inline uint128 operator>>(const uint128& val, int amount) {
  // uint64_t shifts of >= 64 are undefined, so we will need some
  // special-casing.
  if (amount < 64) {
    if (amount == 0) {
      return val;
    }
    uint64_t new_hi = Uint128High64(val) >> amount;
    uint64_t new_lo =
        (Uint128Low64(val) >> amount) | (Uint128High64(val) << (64 - amount));
    return uint128(new_hi, new_lo);
  } else if (amount < 128) {
    return uint128(0, Uint128High64(val) >> (amount - 64));
  } else {
    return uint128(0, 0);
  }
}

inline uint128& operator<<=(uint128& self, int amount) {
  // uint64_t shifts of >= 64 are undefined, so we will need some
  // special-casing.
  if (amount < 64) {
    if (amount != 0) {
      self.hi_ = (self.hi_ << amount) | (self.lo_ >> (64 - amount));
      self.lo_ = self.lo_ << amount;
    }
  } else if (amount < 128) {
    self.hi_ = self.lo_ << (amount - 64);
    self.lo_ = 0;
  } else {
    self.hi_ = 0;
    self.lo_ = 0;
  }
  return self;
}

inline uint128& uint128::operator>>=(int amount) {
  // uint64_t shifts of >= 64 are undefined, so we will need some
  // special-casing.
  if (amount < 64) {
    if (amount != 0) {
      lo_ = (lo_ >> amount) | (hi_ << (64 - amount));
      hi_ = hi_ >> amount;
    }
  } else if (amount < 128) {
    lo_ = hi_ >> (amount - 64);
    hi_ = 0;
  } else {
    lo_ = 0;
    hi_ = 0;
  }
  return *this;
}

inline uint128 operator+(const uint128& lhs, const uint128& rhs) {
  return uint128(lhs) += rhs;
}

inline uint128 operator-(const uint128& lhs, const uint128& rhs) {
  return uint128(lhs) -= rhs;
}

inline uint128 operator*(const uint128& lhs, const uint128& rhs) {
  return uint128(lhs) *= rhs;
}

inline uint128 operator/(const uint128& lhs, const uint128& rhs) {
  return uint128(lhs) /= rhs;
}

inline uint128 operator%(const uint128& lhs, const uint128& rhs) {
  return uint128(lhs) %= rhs;
}

inline uint128& uint128::operator+=(const uint128& b) {
  hi_ += b.hi_;
  uint64_t lolo = lo_ + b.lo_;
  if (lolo < lo_)
    ++hi_;
  lo_ = lolo;
  return *this;
}

inline uint128& uint128::operator-=(const uint128& b) {
  hi_ -= b.hi_;
  if (b.lo_ > lo_)
    --hi_;
  lo_ -= b.lo_;
  return *this;
}

inline uint128& uint128::operator*=(const uint128& b) {
  uint64_t a96 = hi_ >> 32;
  uint64_t a64 = hi_ & 0xffffffffu;
  uint64_t a32 = lo_ >> 32;
  uint64_t a00 = lo_ & 0xffffffffu;
  uint64_t b96 = b.hi_ >> 32;
  uint64_t b64 = b.hi_ & 0xffffffffu;
  uint64_t b32 = b.lo_ >> 32;
  uint64_t b00 = b.lo_ & 0xffffffffu;
  // multiply [a96 .. a00] x [b96 .. b00]
  // terms higher than c96 disappear off the high side
  // terms c96 and c64 are safe to ignore carry bit
  uint64_t c96 = a96 * b00 + a64 * b32 + a32 * b64 + a00 * b96;
  uint64_t c64 = a64 * b00 + a32 * b32 + a00 * b64;
  this->hi_ = (c96 << 32) + c64;
  this->lo_ = 0;
  // add terms after this one at a time to capture carry
  *this += uint128(a32 * b00) << 32;
  *this += uint128(a00 * b32) << 32;
  *this += a00 * b00;
  return *this;
}

inline uint128 uint128::operator++(int) {
  uint128 tmp(*this);
  *this += 1;
  return tmp;
}

inline uint128 uint128::operator--(int) {
  uint128 tmp(*this);
  *this -= 1;
  return tmp;
}

inline uint128& uint128::operator++() {
  *this += 1;
  return *this;
}

inline uint128& uint128::operator--() {
  *this -= 1;
  return *this;
}

} // namespace c10

#pragma once

#include <cstdint>
#include <limits>

#if (!defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) || \
     __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "Only little endian supported"
#endif

namespace lazy_tensors {

class alignas(unsigned __int128) uint128 {
 public:
  uint128() = default;

  constexpr uint128(int v)
      : lo_{static_cast<uint64_t>(v)},
        hi_{v < 0 ? (std::numeric_limits<uint64_t>::max)() : 0} {}

  constexpr uint128(long v)
      : lo_{static_cast<uint64_t>(v)},
        hi_{v < 0 ? (std::numeric_limits<uint64_t>::max)() : 0} {}

  constexpr uint128(unsigned long v) : lo_{v}, hi_{0} {}

  constexpr uint128(unsigned __int128 v)
      : lo_{static_cast<uint64_t>(v & ~uint64_t{0})},
        hi_{static_cast<uint64_t>(v >> 64)} {}

  constexpr explicit operator unsigned __int128() const;

  uint128& operator*=(uint128 other);

  uint128& operator^=(uint128 other) {
    hi_ ^= other.hi_;
    lo_ ^= other.lo_;
    return *this;
  }

  // Uint128Low64()
  //
  // Returns the lower 64-bit value of a `uint128` value.
  friend constexpr uint64_t Uint128Low64(uint128 v);

  // Uint128High64()
  //
  // Returns the higher 64-bit value of a `uint128` value.
  friend constexpr uint64_t Uint128High64(uint128 v);

  // MakeUInt128()
  //
  // Constructs a `uint128` numeric value from two 64-bit unsigned integers.
  // Note that this factory function is the only way to construct a `uint128`
  // from integer values greater than 2^64.
  //
  // Example:
  //
  //   lazy_tensors::uint128 big = lazy_tensors::MakeUint128(1, 0);
  friend constexpr uint128 MakeUint128(uint64_t high, uint64_t low);

 private:
  constexpr uint128(uint64_t high, uint64_t low) : lo_{low}, hi_{high} {}

  uint64_t lo_;
  uint64_t hi_;
};

constexpr uint128 MakeUint128(uint64_t high, uint64_t low) {
  return uint128(high, low);
}

constexpr uint128::operator unsigned __int128() const {
  return (static_cast<unsigned __int128>(hi_) << 64) + lo_;
}

// Arithmetic operators.

inline uint128 operator<<(uint128 lhs, int amount) {
  return static_cast<unsigned __int128>(lhs) << amount;
}

inline uint128 operator>>(uint128 lhs, int amount) {
  return static_cast<unsigned __int128>(lhs) >> amount;
}

inline uint128 operator+(uint128 lhs, uint128 rhs) {
  uint128 result = MakeUint128(Uint128High64(lhs) + Uint128High64(rhs),
                               Uint128Low64(lhs) + Uint128Low64(rhs));
  if (Uint128Low64(result) < Uint128Low64(lhs)) {  // check for carry
    return MakeUint128(Uint128High64(result) + 1, Uint128Low64(result));
  }
  return result;
}

inline uint128 operator*(uint128 lhs, uint128 rhs) {
  return static_cast<unsigned __int128>(lhs) *
         static_cast<unsigned __int128>(rhs);
}

inline uint128& uint128::operator*=(uint128 other) {
  *this = *this * other;
  return *this;
}

constexpr uint64_t Uint128Low64(uint128 v) { return v.lo_; }

constexpr uint64_t Uint128High64(uint128 v) { return v.hi_; }

// Comparison operators.

inline bool operator==(uint128 lhs, uint128 rhs) {
  return (Uint128Low64(lhs) == Uint128Low64(rhs) &&
          Uint128High64(lhs) == Uint128High64(rhs));
}

inline bool operator!=(uint128 lhs, uint128 rhs) { return !(lhs == rhs); }

// Logical operators.

inline uint128 operator^(uint128 lhs, uint128 rhs) {
  return MakeUint128(Uint128High64(lhs) ^ Uint128High64(rhs),
                     Uint128Low64(lhs) ^ Uint128Low64(rhs));
}

}  // namespace lazy_tensors

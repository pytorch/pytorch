#pragma once

namespace c10 {

// This is just a placeholder for QInt Type
struct alignas(4) QInt {
  int32_t bit_;
  QInt() : bit_(8) {}
  QInt(int bit) : bit_(bit) {}
  int32_t get_bit() const { return bit_; }
};

} // namespace c10

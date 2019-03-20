#pragma once
#include <cstdint>

namespace c10 {
struct alignas(1) qint8 {
  uint8_t val_;
  /* implicit */ qint8(uint8_t val=0) : val_(val) {}
  /* inline operator int64_t() const { */
  /*   return static_cast<int64_t>(val_); */
  /* } */

  inline operator uint32_t() const {
    return static_cast<uint32_t>(val_);
  }

  /* inline operator double() const { */
  /*   return static_cast<double>(val_); */
  /* } */

  /* inline operator float() const { */
  /*   return static_cast<float>(val_); */
  /* } */

};

} // namespace c10

#pragma once
#include <cstdint>

namespace c10 {
struct alignas(1) qint8 {
  uint8_t val_;
  /* implicit */ qint8(uint8_t val=0) : val_(val) {}

  inline operator uint32_t() const {
    return static_cast<uint32_t>(val_);
  }

};

} // namespace c10

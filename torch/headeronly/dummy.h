#pragma once

#include <cstdint>

namespace dummy_types {

// Legacy version representation (pre-2.9.0) - only has id
struct Dummy {
  int32_t id;

  explicit Dummy(int32_t id) : id(id) {}

  int32_t get_id() const {
    return id;
  }
};

} // namespace dummy_types

#pragma once

#include <cstdint>

namespace dummy_types {

namespace v2_8 {
// Legacy version representation (pre-2.9.0) - only has id
struct Dummy {
  int32_t id;

  explicit Dummy(int32_t id) : id(id) {}

  int32_t get_id() const {
    return id;
  }
};
} // namespace v2_8

inline namespace v2_9 {
// New version representation (>= 2.9.0) - has foo and id
struct Dummy {
  int8_t foo;
  int32_t id;

  // Constructors for different versions
  explicit Dummy(int32_t id) : foo(1), id(id) {} // Legacy constructor
  explicit Dummy(int8_t foo, int32_t id)
      : foo(foo), id(id) {} // New constructor

  int8_t get_foo() const {
    return foo;
  }
  int32_t get_id() const {
    return id;
  }
};
} // namespace v2_9

} // namespace dummy_types

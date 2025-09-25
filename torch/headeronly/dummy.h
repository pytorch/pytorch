#pragma once

#include <torch/csrc/stable/version.h>
#include <cstdint>

#include <iostream>

// ======================================================================
// DUMMY TYPE FOR TESTING VERSION-AWARE CONVERSIONS WITH INLINE NAMESPACES
// ======================================================================

namespace dummy_types {

// For versions < 2.9.0 - use inline namespace for legacy
#if defined(TORCH_FEATURE_VERSION) && \
    TORCH_FEATURE_VERSION < ((2ULL << 56) | (9ULL << 48))
inline
#endif
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

// For versions >= 2.9.0 - use inline namespace for new version
#if !defined(TORCH_FEATURE_VERSION) || \
    TORCH_FEATURE_VERSION >= ((2ULL << 56) | (9ULL << 48))
inline
#endif
    namespace v2_9 {
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

#pragma once

#include <c10/macros/Export.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <vector>

namespace c10 {

  constexpr size_t NUMBER_OF_DEVICE_CAPABILITIES = NumScalarTypes;

// Generate bitfields for each scalar type
#define DEFINE_SCALAR_TYPE(_1, n) unsigned int has_##n : 1;

struct C10_API DeviceCapability {
  union {
    struct {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)
    };
    uint64_t capability_bits; // Allow direct bit manipulation
  };

  // Default constructor with all capabilities enabled.
  DeviceCapability() : capability_bits((1ULL << NUMBER_OF_DEVICE_CAPABILITIES) - 1) {}

  // Convert capability bits to vector of supported ScalarTypes
  std::vector<ScalarType> getSupportedScalarTypes() const {
    std::vector<ScalarType> supported_types;
    supported_types.reserve(NUMBER_OF_DEVICE_CAPABILITIES);

    // Check each capability bit and add corresponding ScalarType
    #define CHECK_SCALAR_TYPE(_1, n) \
      if (has_##n) { \
        supported_types.push_back(ScalarType::n); \
      }

    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CHECK_SCALAR_TYPE)

    #undef CHECK_SCALAR_TYPE

    return supported_types;
  }

};

#undef DEFINE_SCALAR_TYPE
} // namespace c10

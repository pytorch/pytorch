#pragma once

#include <c10/macros/Export.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <vector>

namespace c10 {

constexpr size_t NUMBER_OF_DEVICE_CAPABILITIES = NumScalarTypes;

// Generate bitfields for each scalar type
#define DEFINE_SCALAR_TYPE(_1, n) unsigned int has_##n : 1;

/**
 * @brief DeviceCapability represents the the common capabilities that all devices should support.
 *
 * This struct provides a compact way to represent the common capabilities that all devices should support.
 * Includes the following capabilities:
 * - Supported data types
 *
 * Purpose
 * - Enable runtime checking of device capability support
 * - Provide efficient storage for capability information
 * - Support dynamic capability queries for device selection
 * - Enable device-specific optimizations based on supported capabilities
 *
 * Contract
 * - Each bitfield represents support for one device capability
 * - Bit value 1 means the capability is supported, 0 means not supported
 * - The struct is initialized with all capabilities enabled by default
 *
 * @note Adding New Capabilities
 *
 * 1. Define the new capability in the `DeviceCapability` struct
 * 2. Update the support of the new capability in each accelerator implementation
 * 3. Create a new method to check if the capability is supported
 * 4. Update to length of NUMBER_OF_DEVICE_CAPABILITIES
 * 5. Add the new capability to the returned PyObject Dictionary
 */
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

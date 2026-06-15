#pragma once

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Export.h>
#include <torch/headeronly/macros/Macros.h>

#include <cstdint>

namespace c10 {

constexpr size_t NUMBER_OF_DEVICE_CAPABILITIES = NumScalarTypes;

// Generate bitfields for each scalar type
#define DEFINE_SCALAR_TYPE(_1, n) unsigned int has_##n : 1;

// Generate enum indices for each scalar type
#define DEFINE_SCALAR_ENUM(_1, name) kIndex_##name,

enum ScalarTypeIndex {
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_ENUM)
};

/**
 * @brief DeviceCapability represents the the common capabilities that all
 * devices should support.
 *
 * This struct provides a compact way to represent the common capabilities that
 * all devices should support. Includes the following capabilities:
 * - Supported data types
 *
 * Purpose
 * - Enable device-specific optimizations based on supported capabilities
 *
 * Contract
 *
 * Supported data types:
 * - Each bitfield represents support for one device capability
 * - Bit value 1 means the capability is supported, 0 means not supported
 * - The struct is initialized with all capabilities enabled by default
 *
 * @note Adding New Capabilities
 *
 * 1. Define the new capability in the `DeviceCapability` struct
 * 2. Update the support of the new capability in each accelerator
 *    implementation
 * 3. Add the new capability to the returned PyObject Dictionary
 */
struct C10_API DeviceCapability {
  union {
    struct {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)
    } supported_scalar_types;
    uint64_t capability_bits; // Allow direct bit manipulation
  } capability_data{};

  // Default constructor with all capabilities enabled.
  DeviceCapability() {
    capability_data.capability_bits =
        ((1ULL << NUMBER_OF_DEVICE_CAPABILITIES) - 1);
  }

  // Iterate supported ScalarTypes without allocating a vector. visitor is
  // invoked once per supported type, so it is intentionally not forwarded.
  template <typename F>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  void forEachSupportedScalarType(F&& visitor) const {
#define VISIT_SCALAR_TYPE(_1, n)                        \
  if (capability_data.supported_scalar_types.has_##n) { \
    visitor(ScalarType::n);                             \
  }

    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(VISIT_SCALAR_TYPE)

#undef VISIT_SCALAR_TYPE
  }
};

#undef DEFINE_SCALAR_ENUM
#undef DEFINE_SCALAR_TYPE
} // namespace c10

// ScalarTypeIndex is an unscoped enum whose enumerators (kIndex_*) are used
// unqualified throughout c10/aten, so it is defined in c10 above; here we
// re-export the type, its enumerators, and DeviceCapability into
// torch::headeronly.
HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::DeviceCapability;
using c10::NUMBER_OF_DEVICE_CAPABILITIES;
using c10::ScalarTypeIndex;
#define ALIAS_SCALAR_ENUM(_1, name) using c10::kIndex_##name;
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(ALIAS_SCALAR_ENUM)
#undef ALIAS_SCALAR_ENUM
HIDDEN_NAMESPACE_END(torch, headeronly)

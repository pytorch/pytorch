#pragma once

#include <c10/core/Device.h>
#include <c10/util/ArrayRef.h>
#include <c10/macros/Export.h>
#include <atomic>

namespace c10 {

/**
 * Detects when operations are running in fallback mode where
 * cross-device tensors are legitimately created
 */
class C10_API FallbackDetector {
public:
    // Check if MPS fallback is currently enabled
    static bool is_mps_fallback_enabled();
    
    // Check if we're currently in a fallback operation context
    static bool is_in_fallback_context();
    
    // Mark entry/exit of fallback context (thread-local)
    static void enter_fallback_context();
    static void exit_fallback_context();
    
    // RAII helper for fallback context
    class FallbackContext {
    public:
        FallbackContext() { FallbackDetector::enter_fallback_context(); }
        ~FallbackContext() { FallbackDetector::exit_fallback_context(); }
    };

private:
    static thread_local bool in_fallback_context_;
};

/**
 * Enhanced device compatibility checker that's aware of fallback scenarios
 */
class C10_API FallbackAwareDeviceChecker {
public:
    // Check if device combination is allowed considering fallback
    static bool are_devices_compatible(
        Device device1, 
        Device device2,
        const char* operation_name = nullptr);
    
    // Check if CPU/MPS mixing is allowed in current context
    static bool is_cpu_mps_mixing_allowed();
    
    // Validate device compatibility with fallback awareness
    static void validate_device_compatibility(
        ArrayRef<Device> devices,
        const char* operation_name);

private:
    static bool is_mps_cpu_compatible_operation(const char* operation_name);
};

} // namespace c10

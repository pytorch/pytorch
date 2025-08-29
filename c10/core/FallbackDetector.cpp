#include <c10/core/FallbackDetector.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <cstdlib>
#include <unordered_set>

namespace c10 {

// Thread-local fallback context tracking
thread_local bool FallbackDetector::in_fallback_context_ = false;

bool FallbackDetector::is_mps_fallback_enabled() {
  // Check environment variable
  const char* env_var = std::getenv("PYTORCH_ENABLE_MPS_FALLBACK");
  return env_var && std::string(env_var) == "1";
}

bool FallbackDetector::is_in_fallback_context() {
  return in_fallback_context_;
}

void FallbackDetector::enter_fallback_context() {
  in_fallback_context_ = true;
}

void FallbackDetector::exit_fallback_context() {
  in_fallback_context_ = false;
}

// Enhanced device compatibility checking
bool FallbackAwareDeviceChecker::are_devices_compatible(
    Device device1,
    Device device2,
    const char* operation_name) {
  // Same device is always compatible
  if (device1 == device2) {
    return true;
  }

  // Different device types
  if (device1.type() != device2.type()) {
    // CPU/MPS mixing cases
    if ((device1.type() == kCPU && device2.type() == kMPS) ||
        (device1.type() == kMPS && device2.type() == kCPU)) {
      // Allow if we're in fallback context
      if (FallbackDetector::is_in_fallback_context()) {
        return true;
      }

      // Allow if MPS fallback is enabled and this is a compatible operation
      if (FallbackDetector::is_mps_fallback_enabled() &&
          is_mps_cpu_compatible_operation(operation_name)) {
        return true;
      }
    }

    // All other cross-device type combinations are incompatible
    return false;
  }

  // Same device type, different indices (e.g., cuda:0 vs cuda:1)
  // Generally not compatible unless specifically allowed
  return false;
}

bool FallbackAwareDeviceChecker::is_cpu_mps_mixing_allowed() {
  return FallbackDetector::is_in_fallback_context() ||
      FallbackDetector::is_mps_fallback_enabled();
}

void FallbackAwareDeviceChecker::validate_device_compatibility(
    ArrayRef<Device> devices,
    const char* operation_name) {
  if (devices.size() < 2) {
    return; // Nothing to check
  }

  Device primary_device = devices[0];

  for (size_t i = 1; i < devices.size(); ++i) {
    if (!are_devices_compatible(primary_device, devices[i], operation_name)) {
      // Special handling for CPU/MPS to provide helpful error
      if ((primary_device.type() == kCPU && devices[i].type() == kMPS) ||
          (primary_device.type() == kMPS && devices[i].type() == kCPU)) {
        TORCH_CHECK(
            false,
            operation_name,
            ": Expected all tensors to be on the same device, "
            "but found tensors on ",
            primary_device,
            " and ",
            devices[i],
            ". "
            "For MPS operations, consider enabling fallback with "
            "PYTORCH_ENABLE_MPS_FALLBACK=1 or move tensors to the same device using .to()");
      } else {
        TORCH_CHECK(
            false,
            operation_name,
            ": Expected all tensors to be on the same device, "
            "but found tensors on ",
            primary_device,
            " and ",
            devices[i],
            ". "
            "Consider using .to() to move tensors to the same device.");
      }
    }
  }
}

bool FallbackAwareDeviceChecker::is_mps_cpu_compatible_operation(
    const char* operation_name) {
  if (!operation_name)
    return false;

  std::string op_name(operation_name);
  
  // Operations that commonly use fallback and should allow CPU/MPS mixing
  static const std::unordered_set<std::string> compatible_ops = {
      "lcm",
      "gcd", 
      "bitwise_and",
      "bitwise_or",
      "bitwise_xor",
      "_copy_from_and_resize",
      "copy_",
      "embedding_dense_backward",
      "linalg_solve_triangular",
      // Add more operations that legitimately use fallback
  };

  // Check direct operation name
  if (compatible_ops.find(op_name) != compatible_ops.end()) {
    return true;
  }
  
  // Check for wrapper names (e.g., "wrapper_MPS__linalg_solve_triangular")
  for (const auto& compat_op : compatible_ops) {
    if (op_name.find(compat_op) != std::string::npos) {
      return true;
    }
  }

  return false;
}

} // namespace c10

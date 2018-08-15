#pragma once

// This is directly synchronized with caffe2/proto/caffe2.proto, but
// doesn't require me to figure out how to get Protobuf headers into
// ATen/core (which would require a lot more build system hacking.)
// If you modify me, keep me synchronized with that file.

#include <ATen/core/Macros.h>

#include <ostream>

namespace at {

// Underlying type declared to be int32_t for consistency with protobufs.
enum class DeviceType : int32_t {
  CPU = 0,
  CUDA = 1, // CUDA.
  MKLDNN = 2, // Reserved for explicit MKLDNN
  OPENGL = 3, // OpenGL
  OPENCL = 4, // OpenCL
  IDEEP = 5, // IDEEP.
  HIP = 6, // AMD HIP
  // Change the following number if you add more devices in the code.
  COMPILE_TIME_MAX_DEVICE_TYPES = 7,
  ONLY_FOR_TEST = 20901701, // This device type is only for test.
};

AT_CORE_API std::string DeviceTypeName(
    at::DeviceType d,
    bool lower_case = false);

AT_CORE_API std::ostream& operator<<(std::ostream& stream, at::DeviceType type);

} // namespace at

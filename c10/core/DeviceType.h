#pragma once

// This is directly synchronized with caffe2/proto/caffe2.proto, but
// doesn't require me to figure out how to get Protobuf headers into
// ATen/core (which would require a lot more build system hacking.)
// If you modify me, keep me synchronized with that file.

#include <c10/macros/Macros.h>

#include <ostream>
#include <functional>

namespace c10 {

enum class DeviceType : int16_t {
  CPU = 0,
  CUDA = 1, // CUDA.
  MKLDNN = 2, // Reserved for explicit MKLDNN
  OPENGL = 3, // OpenGL
  OPENCL = 4, // OpenCL
  IDEEP = 5, // IDEEP.
  HIP = 6, // AMD HIP
  FPGA = 7, // FPGA
  MSNPU = 8, // MSNPU
  XLA = 9, // XLA / TPU
  // NB: If you add more devices:
  //  - Change the implementations of DeviceTypeName and isValidDeviceType
  //    in DeviceType.cpp
  //  - Change the number below
  COMPILE_TIME_MAX_DEVICE_TYPES = 10,
  ONLY_FOR_TEST = 20901, // This device type is only for test.
};

// define explicit int constant
constexpr int COMPILE_TIME_MAX_DEVICE_TYPES =
    static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
C10_API std::string DeviceTypeName(
    DeviceType d,
    bool lower_case = false);

C10_API bool isValidDeviceType(DeviceType d);

C10_API std::ostream& operator<<(std::ostream& stream, DeviceType type);

} // namespace at

namespace std {
template <> struct hash<c10::DeviceType> {
  std::size_t operator()(c10::DeviceType k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std

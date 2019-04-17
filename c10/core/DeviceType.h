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

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kHIP = DeviceType::HIP;
constexpr DeviceType kMSNPU = DeviceType::MSNPU;
constexpr DeviceType kXLA = DeviceType::XLA;

// define explicit int constant
constexpr int COMPILE_TIME_MAX_DEVICE_TYPES =
    static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);

static_assert(COMPILE_TIME_MAX_DEVICE_TYPES <= 16,
    "Hey!  You seem to be adding a lot of new DeviceTypes.  The intent was "
    "for this constant to reflect the actual number of DeviceTypes we support "
    "in PyTorch; it's important that this number is not too large as we "
    "use this to allocate stack arrays in some places in our code.  If you "
    "are indeed just adding the 17th device type, feel free to change "
    "the check to 32; but if you are adding some sort of extensible device "
    "types registration, please be aware that you are affecting code that "
    "this number is small.  Try auditing uses of this constant.");

C10_API std::string DeviceTypeName(
    DeviceType d,
    bool lower_case = false);

C10_API bool isValidDeviceType(DeviceType d);

C10_API std::ostream& operator<<(std::ostream& stream, DeviceType type);

} // namespace c10

namespace std {
template <> struct hash<c10::DeviceType> {
  std::size_t operator()(c10::DeviceType k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std

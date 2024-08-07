#pragma once

// This is directly synchronized with caffe2/proto/caffe2.proto, but
// doesn't require me to figure out how to get Protobuf headers into
// ATen/core (which would require a lot more build system hacking.)
// If you modify me, keep me synchronized with that file.

#include <c10/macros/Export.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>

namespace c10 {

// These contains all device types that also have a BackendComponent
// and therefore participate in per-backend functionality dispatch keys.
// This is most backends except PrivateUse2 and PrivateUse3
#define C10_FORALL_BACKEND_DEVICE_TYPES(_, extra) \
  _(CPU, extra)                                   \
  _(CUDA, extra)                                  \
  _(HIP, extra)                                   \
  _(XLA, extra)                                   \
  _(MPS, extra)                                   \
  _(IPU, extra)                                   \
  _(XPU, extra)                                   \
  _(HPU, extra)                                   \
  _(VE, extra)                                    \
  _(Lazy, extra)                                  \
  _(Meta, extra)                                  \
  _(MTIA, extra)                                  \
  _(PrivateUse1, extra)

enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1, // CUDA.
  MKLDNN = 2, // Reserved for explicit MKLDNN
  OPENGL = 3, // OpenGL
  OPENCL = 4, // OpenCL
  IDEEP = 5, // IDEEP.
  HIP = 6, // AMD HIP
  FPGA = 7, // FPGA
  MAIA = 8, // ONNX Runtime / Microsoft
  XLA = 9, // XLA / TPU
  Vulkan = 10, // Vulkan
  Metal = 11, // Metal
  XPU = 12, // XPU
  MPS = 13, // MPS
  Meta = 14, // Meta (tensors with no data)
  HPU = 15, // HPU / HABANA
  VE = 16, // SX-Aurora / NEC
  Lazy = 17, // Lazy Tensors
  IPU = 18, // Graphcore IPU
  MTIA = 19, // Meta training and inference devices
  PrivateUse1 = 20, // PrivateUse1 device
  // NB: If you add more devices:
  //  - Change the implementations of DeviceTypeName and isValidDeviceType
  //    in DeviceType.cpp
  //  - Change the number below
  COMPILE_TIME_MAX_DEVICE_TYPES = 21,
};

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kHIP = DeviceType::HIP;
constexpr DeviceType kFPGA = DeviceType::FPGA;
constexpr DeviceType kMAIA = DeviceType::MAIA;
constexpr DeviceType kXLA = DeviceType::XLA;
constexpr DeviceType kMPS = DeviceType::MPS;
constexpr DeviceType kMeta = DeviceType::Meta;
constexpr DeviceType kVulkan = DeviceType::Vulkan;
constexpr DeviceType kMetal = DeviceType::Metal;
constexpr DeviceType kXPU = DeviceType::XPU;
constexpr DeviceType kHPU = DeviceType::HPU;
constexpr DeviceType kVE = DeviceType::VE;
constexpr DeviceType kLazy = DeviceType::Lazy;
constexpr DeviceType kIPU = DeviceType::IPU;
constexpr DeviceType kMTIA = DeviceType::MTIA;
constexpr DeviceType kPrivateUse1 = DeviceType::PrivateUse1;

// define explicit int constant
constexpr int COMPILE_TIME_MAX_DEVICE_TYPES =
    static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);

static_assert(
    COMPILE_TIME_MAX_DEVICE_TYPES <= 21,
    "Hey!  You seem to be adding a lot of new DeviceTypes.  The intent was "
    "for this constant to reflect the actual number of DeviceTypes we support "
    "in PyTorch; it's important that this number is not too large as we "
    "use this to allocate stack arrays in some places in our code.  If you "
    "are indeed just adding the 20th device type, feel free to change "
    "the check to 32; but if you are adding some sort of extensible device "
    "types registration, please be aware that you are affecting code that "
    "this number is small.  Try auditing uses of this constant.");

C10_API std::string DeviceTypeName(DeviceType d, bool lower_case = false);

C10_API bool isValidDeviceType(DeviceType d);

C10_API std::ostream& operator<<(std::ostream& stream, DeviceType type);

C10_API void register_privateuse1_backend(const std::string& backend_name);
C10_API std::string get_privateuse1_backend(bool lower_case = true);

C10_API bool is_privateuse1_backend_registered();

} // namespace c10

namespace std {
template <>
struct hash<c10::DeviceType> {
  std::size_t operator()(c10::DeviceType k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std

namespace torch {
// NOLINTNEXTLINE(misc-unused-using-decls)
using c10::DeviceType;
} // namespace torch

#pragma once

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <string>

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

// Need to follow C10_API's behavior here
// TODO: refactor part of c10/macros/Export.h to torch/standalone
#ifdef C10_BUILD_MAIN_LIB
#define TORCH_STANDALONE_API C10_EXPORT
#else
#define TORCH_STANDALONE_API C10_IMPORT
#endif

namespace torch::standalone {
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
  //  - Change the number below
  COMPILE_TIME_MAX_DEVICE_TYPES = 21,
};

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kMKLDNN = DeviceType::MKLDNN;
constexpr DeviceType kOPENGL = DeviceType::OPENGL;
constexpr DeviceType kOPENCL = DeviceType::OPENCL;
constexpr DeviceType kIDEEP = DeviceType::IDEEP;
constexpr DeviceType kHIP = DeviceType::HIP;
constexpr DeviceType kFPGA = DeviceType::FPGA;
constexpr DeviceType kMAIA = DeviceType::MAIA;
constexpr DeviceType kXLA = DeviceType::XLA;
constexpr DeviceType kVulkan = DeviceType::Vulkan;
constexpr DeviceType kMetal = DeviceType::Metal;
constexpr DeviceType kXPU = DeviceType::XPU;
constexpr DeviceType kMPS = DeviceType::MPS;
constexpr DeviceType kMeta = DeviceType::Meta;
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

// NB: Per the C++ standard (e.g.,
// https://stackoverflow.com/questions/18195312/what-happens-if-you-static-cast-invalid-value-to-enum-class)
// as long as you cast from the same underlying type, it is always valid to cast
// into an enum class (even if the value would be invalid by the enum.)  Thus,
// the caller is allowed to cast a possibly invalid int16_t to DeviceType and
// then pass it to this function.  (I considered making this function take an
// int16_t directly, but that just seemed weird.)
inline TORCH_STANDALONE_API bool isValidDeviceType(DeviceType d) {
  switch (d) {
    case DeviceType::CPU:
    case DeviceType::CUDA:
    case DeviceType::OPENGL:
    case DeviceType::OPENCL:
    case DeviceType::MKLDNN:
    case DeviceType::IDEEP:
    case DeviceType::HIP:
    case DeviceType::VE:
    case DeviceType::FPGA:
    case DeviceType::MAIA:
    case DeviceType::XLA:
    case DeviceType::Lazy:
    case DeviceType::MPS:
    case DeviceType::Vulkan:
    case DeviceType::Metal:
    case DeviceType::XPU:
    case DeviceType::Meta:
    case DeviceType::HPU:
    case DeviceType::IPU:
    case DeviceType::MTIA:
    case DeviceType::PrivateUse1:
      return true;
    default:
      return false;
  }
}

#ifdef TORCH_STANDALONE
// The standalone mode doesn't support register_privateuse1_backend
inline TORCH_STANDALONE_API std::string get_privateuse1_backend(bool lower_case = true) {
    return lower_case ? "privateuse1" : "PrivateUse1";
}
#else
TORCH_STANDALONE_API std::string get_privateuse1_backend(bool lower_case = true);
#endif

inline TORCH_STANDALONE_API std::string DeviceTypeName(DeviceType d, bool lower_case = false) {
  switch (d) {
    // I considered instead using ctype::tolower to lower-case the strings
    // on the fly, but this seemed a bit much.
    case DeviceType::CPU:
      return lower_case ? "cpu" : "CPU";
    case DeviceType::CUDA:
      return lower_case ? "cuda" : "CUDA";
    case DeviceType::OPENGL:
      return lower_case ? "opengl" : "OPENGL";
    case DeviceType::OPENCL:
      return lower_case ? "opencl" : "OPENCL";
    case DeviceType::MKLDNN:
      return lower_case ? "mkldnn" : "MKLDNN";
    case DeviceType::IDEEP:
      return lower_case ? "ideep" : "IDEEP";
    case DeviceType::HIP:
      return lower_case ? "hip" : "HIP";
    case DeviceType::VE:
      return lower_case ? "ve" : "VE";
    case DeviceType::FPGA:
      return lower_case ? "fpga" : "FPGA";
    case DeviceType::MAIA:
      return lower_case ? "maia" : "MAIA";
    case DeviceType::XLA:
      return lower_case ? "xla" : "XLA";
    case DeviceType::Lazy:
      return lower_case ? "lazy" : "LAZY";
    case DeviceType::MPS:
      return lower_case ? "mps" : "MPS";
    case DeviceType::Vulkan:
      return lower_case ? "vulkan" : "VULKAN";
    case DeviceType::Metal:
      return lower_case ? "metal" : "METAL";
    case DeviceType::XPU:
      return lower_case ? "xpu" : "XPU";
    case DeviceType::Meta:
      return lower_case ? "meta" : "META";
    case DeviceType::HPU:
      return lower_case ? "hpu" : "HPU";
    case DeviceType::IPU:
      return lower_case ? "ipu" : "IPU";
    case DeviceType::MTIA:
      return lower_case ? "mtia" : "MTIA";
    case DeviceType::PrivateUse1:
      return get_privateuse1_backend(lower_case);
    default:
      TORCH_CHECK(
          false,
          "Unknown device: ",
          static_cast<int16_t>(d),
          ". If you have recently updated the caffe2.proto file to add a new "
          "device type, did you forget to update the DeviceTypeName() "
          "function to reflect such recent changes?");
      // The below code won't run but is needed to suppress some compiler
      // warnings.
      return "";
  }
}

inline TORCH_STANDALONE_API std::ostream& operator<<(std::ostream& stream, DeviceType type) {
  stream << DeviceTypeName(type, /* lower case */ true);
  return stream;
}
} // namespace torch::standalone

namespace std {
template <>
struct hash<torch::standalone::DeviceType> {
  std::size_t operator()(torch::standalone::DeviceType k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std

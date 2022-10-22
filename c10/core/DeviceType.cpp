#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <atomic>
#include <memory>
#include <mutex>

namespace c10 {

std::string DeviceTypeName(DeviceType d, bool lower_case) {
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
    case DeviceType::ORT:
      return lower_case ? "ort" : "ORT";
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
    case DeviceType::PrivateUse1:
      return get_privateuse1_backend(/*lowercase=*/lower_case);
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

// NB: Per the C++ standard (e.g.,
// https://stackoverflow.com/questions/18195312/what-happens-if-you-static-cast-invalid-value-to-enum-class)
// as long as you cast from the same underlying type, it is always valid to cast
// into an enum class (even if the value would be invalid by the enum.)  Thus,
// the caller is allowed to cast a possibly invalid int16_t to DeviceType and
// then pass it to this function.  (I considered making this function take an
// int16_t directly, but that just seemed weird.)
bool isValidDeviceType(DeviceType d) {
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
    case DeviceType::ORT:
    case DeviceType::XLA:
    case DeviceType::Lazy:
    case DeviceType::MPS:
    case DeviceType::Vulkan:
    case DeviceType::Metal:
    case DeviceType::XPU:
    case DeviceType::Meta:
    case DeviceType::HPU:
    case DeviceType::IPU:
    case DeviceType::PrivateUse1:
      return true;
    default:
      return false;
  }
}

std::ostream& operator<<(std::ostream& stream, DeviceType type) {
  stream << DeviceTypeName(type, /* lower case */ true);
  return stream;
}

// We use both a mutex and an atomic here because:
// (1) Mutex is needed during writing:
//     We need to first check the value and potentially error,
//     before setting the value (without any one else racing in the middle).
//     It's also totally fine for this to be slow, since it happens exactly once
//     at import time.
// (2) Atomic is needed during reading:
//     Whenever a user prints a privatuse1 device name, they need to read this
//     variable. Although unlikely, we'll data race if someone else is trying to
//     set this variable at the same time that another thread is print the
//     device name. We could re-use the same mutex, but reading the atomic will
//     be much faster.
static std::atomic<bool> privateuse1_backend_name_set;
static std::string privateuse1_backend_name;
static std::mutex privateuse1_lock;

std::string get_privateuse1_backend(bool lower_case) {
  // Applying the same atomic read memory ordering logic as in Note [Memory
  // ordering on Python interpreter tag].
  auto name_registered =
      privateuse1_backend_name_set.load(std::memory_order_acquire);
  // Guaranteed that if the flag is set, then privateuse1_backend_name has been
  // set, and will never be written to.
  auto backend_name =
      name_registered ? privateuse1_backend_name : "privateuseone";
  return backend_name;
}

void register_privateuse1_backend(std::string backend_name) {
  std::lock_guard<std::mutex> guard(privateuse1_lock);
  TORCH_CHECK(
      !privateuse1_backend_name_set.load() ||
          privateuse1_backend_name == backend_name,
      "torch.register_privateuse1_backend() has already been set! Current backend: ",
      privateuse1_backend_name);

  privateuse1_backend_name = backend_name;
  // Invariant: once this flag is set, privateuse1_backend_name is NEVER written
  // to.
  privateuse1_backend_name_set.store(true, std::memory_order_relaxed);
}

} // namespace c10

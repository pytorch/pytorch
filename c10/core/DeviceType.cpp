#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <mutex>
#include <optional>

namespace c10 {
namespace {

constexpr std::array<DeviceType, 3> kPrivateUseDeviceTypes = {
    DeviceType::PrivateUse1,
    DeviceType::PrivateUse2,
    DeviceType::PrivateUse3};

constexpr std::array<const char*, 3> kPrivateUseDefaultBackendNames = {
    "privateuseone",
    "privateusetwo",
    "privateusethree"};

std::optional<size_t> privateuse_backend_index(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::PrivateUse1:
      return 0;
    case DeviceType::PrivateUse2:
      return 1;
    case DeviceType::PrivateUse3:
      return 2;
    default:
      return std::nullopt;
  }
}

bool is_reserved_privateuse_backend_name(const std::string& backend_name) {
  static const std::array<std::string, 6> types = {
      "cpu", "cuda", "hip", "mps", "xpu", "mtia"};
  return std::find(types.begin(), types.end(), backend_name) != types.end();
}

bool is_in_tree_device_name(const std::string& backend_name) {
  static const std::array<std::string, 20> types = {
      "cpu",   "cuda", "mkldnn", "opengl", "opencl", "ideep", "hip",
      "ve",    "fpga", "maia",   "xla",    "lazy",   "mps",   "vulkan",
      "metal", "xpu",  "meta",   "hpu",    "ipu",    "mtia"};
  return std::find(types.begin(), types.end(), backend_name) != types.end();
}

static std::array<std::atomic<bool>, 3> privateuse_backend_name_set{};
static std::array<std::string, 3> privateuse_backend_names;
static std::mutex privateuse_lock;

void register_privateuse_backend_locked(
    DeviceType device_type,
    const std::string& backend_name,
    const char* api_name) {
  const auto idx = privateuse_backend_index(device_type);
  TORCH_CHECK(
      idx, "Expected a private-use device type, but got: ", device_type);

  TORCH_CHECK(
      !privateuse_backend_name_set[*idx].load() ||
          privateuse_backend_names[*idx] == backend_name,
      "torch.",
      api_name,
      "() has already been set! Current backend: ",
      privateuse_backend_names[*idx]);

  TORCH_CHECK(
      !is_reserved_privateuse_backend_name(backend_name),
      "Cannot register privateuse backend with in-tree device name: ",
      backend_name);

  for (size_t i = 0; i < kPrivateUseDeviceTypes.size(); ++i) {
    TORCH_CHECK(
        i == *idx || backend_name != kPrivateUseDefaultBackendNames[i],
        "Cannot register privateuse backend with another private-use slot name: ",
        backend_name);
    TORCH_CHECK(
        i == *idx || !privateuse_backend_name_set[i].load() ||
            privateuse_backend_names[i] != backend_name,
        "Private-use backend has already been registered with device type ",
        kPrivateUseDeviceTypes[i],
        ": ",
        backend_name);
  }

  if (privateuse_backend_name_set[*idx].load()) {
    return;
  }
  privateuse_backend_names[*idx] = backend_name;
  // Invariant: once this flag is set, privateuse_backend_names[idx] is NEVER
  // written to again.
  privateuse_backend_name_set[*idx].store(true, std::memory_order_release);
}

} // namespace

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
    case DeviceType::PrivateUse2:
    case DeviceType::PrivateUse3:
      return get_privateuse_backend(d, /*lower_case=*/lower_case);
    default:
      TORCH_CHECK(
          false,
          "Unknown device: ",
          static_cast<int16_t>(d),
          ". If you have recently updated the caffe2.proto file to add a new "
          "device type, did you forget to update the DeviceTypeName() "
          "function to reflect such recent changes?");
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
    case DeviceType::PrivateUse2:
    case DeviceType::PrivateUse3:
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
//     Whenever a user prints a privateuse1 device name, they need to read this
//     variable. Although unlikely, we'll data race if someone else is trying to
//     set this variable at the same time that another thread is print the
//     device name. We could reuse the same mutex, but reading the atomic will
//     be much faster.
std::string get_privateuse_backend(DeviceType device_type, bool lower_case) {
  const auto idx = privateuse_backend_index(device_type);
  TORCH_CHECK(
      idx, "Expected a private-use device type, but got: ", device_type);
  // Applying the same atomic read memory ordering logic as in Note [Memory
  // ordering on Python interpreter tag].
  auto name_registered =
      privateuse_backend_name_set[*idx].load(std::memory_order_acquire);
  // Guaranteed that if the flag is set, then privateuse_backend_names[idx] has
  // been set, and will never be written to.
  auto backend_name = name_registered ? privateuse_backend_names[*idx]
                                      : kPrivateUseDefaultBackendNames[*idx];
  auto op_case = lower_case ? ::tolower : ::toupper;
  std::ranges::transform(backend_name, backend_name.begin(), op_case);
  return backend_name;
}

std::string get_privateuse1_backend(bool lower_case) {
  return get_privateuse_backend(DeviceType::PrivateUse1, lower_case);
}

void register_privateuse1_backend(const std::string& backend_name) {
  std::lock_guard<std::mutex> guard(privateuse_lock);
  register_privateuse_backend_locked(
      DeviceType::PrivateUse1, backend_name, "register_privateuse1_backend");
}

DeviceType register_privateuse_backend(const std::string& backend_name) {
  std::lock_guard<std::mutex> guard(privateuse_lock);
  for (size_t i = 0; i < kPrivateUseDeviceTypes.size(); ++i) {
    if (privateuse_backend_name_set[i].load() &&
        privateuse_backend_names[i] == backend_name) {
      return kPrivateUseDeviceTypes[i];
    }
  }

  for (size_t i = 0; i < kPrivateUseDeviceTypes.size(); ++i) {
    if (backend_name == kPrivateUseDefaultBackendNames[i]) {
      register_privateuse_backend_locked(
          kPrivateUseDeviceTypes[i],
          backend_name,
          "register_privateuse_backend");
      return kPrivateUseDeviceTypes[i];
    }
  }

  TORCH_CHECK(
      !is_in_tree_device_name(backend_name),
      "Cannot register privateuse backend with in-tree device name: ",
      backend_name);

  for (size_t i = 0; i < kPrivateUseDeviceTypes.size(); ++i) {
    if (!privateuse_backend_name_set[i].load()) {
      register_privateuse_backend_locked(
          kPrivateUseDeviceTypes[i],
          backend_name,
          "register_privateuse_backend");
      return kPrivateUseDeviceTypes[i];
    }
  }

  TORCH_CHECK(
      false,
      "No private-use backend slots are available. PyTorch supports at most ",
      kPrivateUseDeviceTypes.size(),
      " registered private-use backends in a single process.");
}

bool is_privateuse1_backend_registered() {
  return is_privateuse_backend_registered(DeviceType::PrivateUse1);
}

bool is_privateuse_backend(DeviceType device_type) {
  return privateuse_backend_index(device_type).has_value();
}

bool is_privateuse_backend_registered(DeviceType device_type) {
  const auto idx = privateuse_backend_index(device_type);
  TORCH_CHECK(
      idx, "Expected a private-use device type, but got: ", device_type);
  return privateuse_backend_name_set[*idx].load(std::memory_order_acquire);
}

std::optional<DeviceType> get_privateuse_backend_device_type(
    const std::string& backend_name) {
  for (size_t i = 0; i < kPrivateUseDeviceTypes.size(); ++i) {
    if (backend_name == kPrivateUseDefaultBackendNames[i]) {
      return kPrivateUseDeviceTypes[i];
    }
  }
  for (size_t i = 0; i < kPrivateUseDeviceTypes.size(); ++i) {
    if (privateuse_backend_name_set[i].load(std::memory_order_acquire) &&
        get_privateuse_backend(kPrivateUseDeviceTypes[i]) == backend_name) {
      return kPrivateUseDeviceTypes[i];
    }
  }
  return std::nullopt;
}

} // namespace c10

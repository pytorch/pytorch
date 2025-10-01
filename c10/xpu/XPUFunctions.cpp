#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/xpu/XPUFunctions.h>

#include <vector>

namespace c10::xpu {
namespace {

/*
 * Note [Device Management]
 *
 * An Intel GPU device qualifies as a type of SYCL device. This classification
 * allows for the runtime querying of Intel GPU device information through the
 * SYCL runtime library.
 *
 * Device status is managed through a SYCL device pool, with SYCL devices
 * determined at runtime. There's currently a SYCL device pool that is lazily
 * created and only initialized once, ensuring thread-local safety. Each device
 * within the device pool shares the same default context.
 *
 * In certain scenarios, GPU devices may reside on separate SYCL platforms. For
 * instance, on Windows, an integrated GPU (iGPU) and a discrete GPU (dGPU) may
 * exist on different platforms. Since sycl::context cannot span across multiple
 * platforms, creating a single default context that includes both becomes
 * infeasible.
 *
 * To address this limitation, we prioritize the enumeration of dGPU. The device
 * enumeration logic is as follows:
 * 1. Identify the first Level Zero (L0) platform that contains at least one
 *    dGPU and enumerate all dGPUs on that platform.
 * 2. If no dGPU is found, identify the first L0 platform containing at least
 *    one iGPU and enumerate all iGPUs on that platform.
 * 3. If neither dGPUs nor iGPUs are found, conclude that no GPUs are available.
 */
c10::once_flag init_flag;
thread_local DeviceIndex curDeviceIndex = 0;

struct DevicePool {
  std::vector<std::unique_ptr<sycl::device>> devices;
  std::unique_ptr<sycl::context> context;
} gDevicePool;

void enumDevices(std::vector<std::unique_ptr<sycl::device>>& devices) {
  // See Note [Device Management] for more details.
  auto platform_list = sycl::platform::get_platforms();
  auto is_igpu = [](const sycl::device& device) {
    // Generally, iGPUs share a unified memory subsystem with the host.
    return device.get_info<sycl::info::device::host_unified_memory>();
  };

  // Check if a platform contains at least one GPU (either iGPU or dGPU).
  auto has_gpu = [&is_igpu](const sycl::platform& platform, bool check_igpu) {
    // Only consider platforms using the Level Zero backend.
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      return false;
    }
    // Check if the platform contains at least one GPU.
    for (const auto& device : platform.get_devices()) {
      if (device.is_gpu() &&
          (check_igpu ? is_igpu(device) : !is_igpu(device))) {
        return true;
      }
    }
    // No GPU found on the platform.
    return false;
  };

  // Case 1: Platform with dGPU found. Most platforms with dGPU only have dGPU
  // or a combination of dGPU and iGPU.
  for (const auto& platform : platform_list) {
    // Find the first platform that contains at least one dGPU.
    if (has_gpu(platform, /*check_igpu=*/false)) {
      for (const auto& device : platform.get_devices()) {
        // Only add all dGPUs to the device list.
        if (device.is_gpu() && !is_igpu(device)) {
          devices.push_back(std::make_unique<sycl::device>(device));
        }
      }
      return; // Exit early since we already found a platform with dGPU.
    }
  }

  // Case 2: No dGPU found, but a platform with iGPU is available.
  for (const auto& platform : platform_list) {
    // Find the first platform that contains at least one iGPU.
    if (has_gpu(platform, /*check_igpu=*/true)) {
      for (const auto& device : platform.get_devices()) {
        // Add all iGPUs to the device list.
        if (device.is_gpu()) { // If the device is a GPU, it must be a iGPU.
          devices.push_back(std::make_unique<sycl::device>(device));
        }
      }
      return; // Exit early since we already found a platform with iGPU.
    }
  }

  // Case 3: No GPUs found (neither dGPU nor iGPU) - Do nothing.
}

inline void initGlobalDevicePoolState() {
  // Attempt to initialize XPU devices. If no device is found or the driver is
  // not installed correctly, issue a warning message instead of raising an
  // exception to avoid disrupting the user experience.
  try {
    // Enumerate all GPU devices and record them.
    enumDevices(gDevicePool.devices);
  } catch (const sycl::exception& e) {
    TORCH_WARN(
        "Failed to initialize XPU devices. The driver may not be installed, installed incorrectly, or incompatible with the current setup. ",
        "Please refer to the guideline (https://github.com/pytorch/pytorch?tab=readme-ov-file#intel-gpu-support) for proper installation and configuration.");
    return;
  }
  if (gDevicePool.devices.empty()) {
    TORCH_WARN("XPU device count is zero!");
    return;
  }
  // Ensures that the number of GPU devices does not exceed the maximum
  // allowable value for DeviceIndex.
  TORCH_CHECK(
      gDevicePool.devices.size() <= std::numeric_limits<DeviceIndex>::max(),
      "Too many XPU devices, DeviceIndex overflowed!");

#if defined(_WIN32) && SYCL_COMPILER_VERSION < 20250000
  // The default context feature is disabled by default on Windows for SYCL
  // compiler versions earlier than 2025.0.0.
  std::vector<sycl::device> deviceList;
  for (auto it = gDevicePool.devices.begin(); it != gDevicePool.devices.end();
       ++it) {
    deviceList.push_back(*(*it));
  }
  gDevicePool.context = std::make_unique<sycl::context>(deviceList);
#else
  // The default context is utilized for each Intel GPU device, allowing the
  // retrieval of the context from any GPU device.
  const auto& platform = gDevicePool.devices[0]->get_platform();
  gDevicePool.context = std::make_unique<sycl::context>(
#if SYCL_COMPILER_VERSION >= 20250200
      platform.khr_get_default_context());
#else
      platform.ext_oneapi_get_default_context());
#endif
#endif
}

inline void initDevicePoolCallOnce() {
  c10::call_once(init_flag, initGlobalDevicePoolState);
}

void initDeviceProperties(DeviceProp* device_prop, DeviceIndex device) {
  using namespace sycl::info;
  using namespace sycl::ext;
  // Get raw sycl device associated with device index.
  auto& raw_device = *gDevicePool.devices[device];

  // Initialize the device properties associated with the specific device.
#define ASSIGN_DEVICE_PROP(property) \
  device_prop->property = raw_device.get_info<device::property>();

#define ASSIGN_EXT_DEVICE_PROP(property, aspect_tag, default_value)            \
  device_prop->property = raw_device.has(sycl::aspect::ext_intel_##aspect_tag) \
      ? raw_device.get_info<intel::info::device::property>()                   \
      : default_value;

#define ASSIGN_DEVICE_ASPECT(member) \
  device_prop->has_##member = raw_device.has(sycl::aspect::member);

#define ASSIGN_EXP_CL_ASPECT(member)                                       \
  device_prop->has_##member = raw_device.ext_oneapi_supports_cl_extension( \
      "cl_intel_" #member, &cl_version);

#define ASSIGN_EXP_DEVICE_PROP(property) \
  device_prop->property =                \
      raw_device.get_info<oneapi::experimental::info::device::property>();

  AT_FORALL_XPU_DEVICE_PROPERTIES(ASSIGN_DEVICE_PROP);

  device_prop->platform_name =
      raw_device.get_info<device::platform>().get_info<platform::name>();

  AT_FORALL_XPU_EXT_DEVICE_PROPERTIES(ASSIGN_EXT_DEVICE_PROP);

  AT_FORALL_XPU_DEVICE_ASPECT(ASSIGN_DEVICE_ASPECT);

  // TODO: Remove cl_version since it is unnecessary.
  sycl::ext::oneapi::experimental::cl_version cl_version;
  AT_FORALL_XPU_EXP_CL_ASPECT(ASSIGN_EXP_CL_ASPECT);

#if SYCL_COMPILER_VERSION >= 20250000
  AT_FORALL_XPU_EXP_DEVICE_PROPERTIES(ASSIGN_EXP_DEVICE_PROP);
#endif

  return;
}

} // anonymous namespace

sycl::device& get_raw_device(DeviceIndex device) {
  initDevicePoolCallOnce();
  check_device_index(device);
  return *gDevicePool.devices[device];
}

sycl::context& get_device_context() {
  initDevicePoolCallOnce();
  TORCH_CHECK(
      gDevicePool.context,
      "Device pool initialization failed, you might not have an XPU device.")
  return *gDevicePool.context;
}

void get_device_properties(DeviceProp* device_prop, DeviceIndex device) {
  initDevicePoolCallOnce();
  TORCH_CHECK(device_prop, "device_prop is an invalid pointer.");
  check_device_index(device);
  initDeviceProperties(device_prop, device);
}

DeviceIndex get_device_idx_from_pointer(void* ptr) {
  initDevicePoolCallOnce();
  TORCH_CHECK(ptr, "ptr is an invalid pointer.");
  auto type = sycl::get_pointer_type(ptr, get_device_context());
  TORCH_CHECK(
      type == sycl::usm::alloc::device, "ptr is not a device type pointer.");

  sycl::device raw_device = sycl::get_pointer_device(ptr, get_device_context());
  auto match_device = [raw_device](const auto& device) -> bool {
    return raw_device == *device;
  };
  auto it = std::find_if(
      gDevicePool.devices.begin(), gDevicePool.devices.end(), match_device);
  TORCH_CHECK(
      it != gDevicePool.devices.end(),
      "Can't find the pointer from XPU devices.");
  return static_cast<DeviceIndex>(
      std::distance(gDevicePool.devices.begin(), it));
}

DeviceIndex device_count() {
  initDevicePoolCallOnce();
  return static_cast<DeviceIndex>(gDevicePool.devices.size());
}

DeviceIndex device_count_ensure_non_zero() {
  auto count = device_count();
  // Zero gpus could produce a warning in `device_count` but we fail here.
  TORCH_CHECK(count, "No XPU devices are available.");
  return count;
}

DeviceIndex current_device() {
  initDevicePoolCallOnce();
  return curDeviceIndex;
}

void set_device(DeviceIndex device) {
  initDevicePoolCallOnce();
  check_device_index(device);
  curDeviceIndex = device;
}

c10::DeviceIndex exchange_device(c10::DeviceIndex to_device) {
  auto cur_device = current_device();
  if (to_device == cur_device) {
    return cur_device;
  }
  set_device(to_device);
  return cur_device;
}

c10::DeviceIndex maybe_exchange_device(c10::DeviceIndex to_device) {
  return exchange_device(to_device);
}

} // namespace c10::xpu

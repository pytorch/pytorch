#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/xpu/XPUFunctions.h>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <cmath>
#include <deque>
#include <mutex>
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
 */
static c10::once_flag init_flag;
static thread_local DeviceIndex curDeviceIndex = 0;

struct DevicePool {
  std::vector<std::unique_ptr<sycl::device>> devices;
  std::unique_ptr<sycl::context> context;
} gDevicePool;

static void enumDevices(std::vector<std::unique_ptr<sycl::device>>& devices) {
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated GPU devices from the specific platform.
  for (const auto& platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        devices.push_back(std::make_unique<sycl::device>(device));
      }
    }
  }
}

static inline int deviceCountImpl(
    std::vector<std::unique_ptr<sycl::device>>& devices) {
  enumDevices(devices);
  return static_cast<int>(devices.size());
}

static inline void initGlobalDevicePoolState() {
  // Get device count and record the all GPU devices.
  auto device_count = deviceCountImpl(gDevicePool.devices);
  if (device_count <= 0) {
    TORCH_WARN("XPU device count is zero!");
    return;
  }

  // The default context is utilized for each Intel GPU device, allowing the
  // retrieval of the context from any GPU device.
  gDevicePool.context = std::make_unique<sycl::context>(
      gDevicePool.devices[0]->get_platform().ext_oneapi_get_default_context());
}

static inline void initDevicePoolCallOnce() {
  c10::call_once(init_flag, initGlobalDevicePoolState);
}

static void initDeviceProperties(DeviceProp* device_prop, int device) {
  using namespace sycl::info;
  using namespace sycl::ext;
  // Get raw sycl device associated with device index.
  auto& raw_device = *gDevicePool.devices[device];

  // clang-format off
  // Initialize the device properties associated with the specific device.
  device_prop->device_name = raw_device.get_info<device::name>();
  device_prop->device_type = raw_device.get_info<device::device_type>();
  device_prop->platform_name = raw_device.get_info<device::platform>().get_info<platform::name>();
  device_prop->vendor = raw_device.get_info<device::vendor>();
  device_prop->driver_version = raw_device.get_info<device::driver_version>();
  device_prop->is_available = raw_device.get_info<device::is_available>();
  device_prop->max_param_size = raw_device.get_info<device::max_parameter_size>();
  device_prop->max_compute_units = raw_device.get_info<device::max_compute_units>();
  device_prop->max_work_item_dims = raw_device.get_info<device::max_work_item_dimensions>();
  device_prop->max_work_group_size = raw_device.get_info<device::max_work_group_size>();
  device_prop->max_num_sub_groups = raw_device.get_info<device::max_num_sub_groups>();
  device_prop->sub_group_sizes = raw_device.get_info<device::sub_group_sizes>();
  device_prop->max_clock_freq = raw_device.get_info<device::max_clock_frequency>();
  device_prop->address_bits = raw_device.get_info<device::address_bits>();
  device_prop->max_mem_alloc_size = raw_device.get_info<device::max_mem_alloc_size>();
  device_prop->mem_base_addr_align = raw_device.get_info<device::mem_base_addr_align>();
  device_prop->half_fp_config = raw_device.get_info<device::half_fp_config>();
  device_prop->single_fp_config = raw_device.get_info<device::single_fp_config>();
  device_prop->double_fp_config = raw_device.get_info<device::double_fp_config>();
  device_prop->global_mem_size = raw_device.get_info<device::global_mem_size>();
  device_prop->global_mem_cache_type = raw_device.get_info<device::global_mem_cache_type>();
  device_prop->global_mem_cache_size = raw_device.get_info<device::global_mem_cache_size>();
  device_prop->global_mem_cache_line_size = raw_device.get_info<device::global_mem_cache_line_size>();
  device_prop->local_mem_type = raw_device.get_info<device::local_mem_type>();
  device_prop->local_mem_size = raw_device.get_info<device::local_mem_size>();
  device_prop->max_sub_devices = raw_device.get_info<device::partition_max_sub_devices>();
  device_prop->profiling_resolution = raw_device.get_info<device::profiling_timer_resolution>();
  device_prop->pref_vec_width_char = raw_device.get_info<device::preferred_vector_width_char>();
  device_prop->pref_vec_width_short = raw_device.get_info<device::preferred_vector_width_short>();
  device_prop->pref_vec_width_int = raw_device.get_info<device::preferred_vector_width_int>();
  device_prop->pref_vec_width_long = raw_device.get_info<device::preferred_vector_width_long>();
  device_prop->pref_vec_width_float = raw_device.get_info<device::preferred_vector_width_float>();
  device_prop->pref_vec_width_double = raw_device.get_info<device::preferred_vector_width_double>();
  device_prop->pref_vec_width_half = raw_device.get_info<device::preferred_vector_width_half>();
  device_prop->native_vec_width_char = raw_device.get_info<device::native_vector_width_char>();
  device_prop->native_vec_width_short = raw_device.get_info<device::native_vector_width_short>();
  device_prop->native_vec_width_int = raw_device.get_info<device::native_vector_width_int>();
  device_prop->native_vec_width_long = raw_device.get_info<device::native_vector_width_long>();
  device_prop->native_vec_width_float = raw_device.get_info<device::native_vector_width_float>();
  device_prop->native_vec_width_double = raw_device.get_info<device::native_vector_width_double>();
  device_prop->native_vec_width_half = raw_device.get_info<device::native_vector_width_half>();

  device_prop->gpu_eu_count = raw_device.has(sycl::aspect::ext_intel_gpu_eu_count)
      ? raw_device.get_info<intel::info::device::gpu_eu_count>()
      : 512;
  device_prop->gpu_eu_count_per_subslice = raw_device.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice)
      ? raw_device.get_info<intel::info::device::gpu_eu_count_per_subslice>()
      : 8;
  device_prop->gpu_eu_simd_width = raw_device.has(sycl::aspect::ext_intel_gpu_eu_simd_width)
      ? raw_device.get_info<intel::info::device::gpu_eu_simd_width>()
      : 8;
  device_prop->gpu_hw_threads_per_eu = raw_device.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)
      ? raw_device.get_info<intel::info::device::gpu_hw_threads_per_eu>()
      : 8;
  // clang-format on
  return;
}

static inline void check_device(int device) {
  int total = static_cast<int>(gDevicePool.devices.size());
  TORCH_CHECK(
      device >= 0 && device < total,
      "device is out of range, device is ",
      device,
      ", total number of device is ",
      total,
      ".");
}

} // anonymous namespace

sycl::device& get_raw_device(int device) {
  initDevicePoolCallOnce();
  check_device(device);
  return *gDevicePool.devices[device];
}

sycl::context& get_device_context() {
  initDevicePoolCallOnce();
  TORCH_CHECK(
      gDevicePool.context,
      "Device pool initialization failed, you might not have an XPU device.")
  return *gDevicePool.context;
}

void get_device_properties(DeviceProp* device_prop, int device) {
  initDevicePoolCallOnce();
  TORCH_CHECK(device_prop, "device_prop is an invalid pointer.");
  check_device(device);
  initDeviceProperties(device_prop, device);
}

int get_device_from_pointer(void* ptr) {
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
      "Cant't find the pointer from XPU devices.");
  return static_cast<int>(std::distance(gDevicePool.devices.begin(), it));
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
  check_device(static_cast<int>(device));
  curDeviceIndex = device;
}

int exchange_device(int to_device) {
  int cur_device = static_cast<int>(current_device());
  if (to_device == cur_device) {
    return cur_device;
  }
  set_device(static_cast<DeviceIndex>(to_device));
  return cur_device;
}

int maybe_exchange_device(int to_device) {
  return c10::xpu::exchange_device(to_device);
}

} // namespace c10::xpu

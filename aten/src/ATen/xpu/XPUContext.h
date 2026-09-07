#pragma once

#include <ATen/Context.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

namespace at::xpu {

// XPU is available if we compiled with XPU.
inline bool is_available() {
  return c10::xpu::device_count() > 0;
}

TORCH_XPU_API DeviceProp* getCurrentDeviceProperties();

TORCH_XPU_API DeviceProp* getDeviceProperties(DeviceIndex device);

TORCH_XPU_API int32_t getGlobalIdxFromDevice(DeviceIndex device);

TORCH_XPU_API bool canDeviceAccessPeer(DeviceIndex device, DeviceIndex peer);

// Returns the maximum number of workitems that are permitted in a work-group
// for a specific kernel on the given device.
template <class KernelClass>
[[nodiscard]] inline size_t getKernelMaxWorkGroupSize(
    DeviceIndex device_index = c10::xpu::current_device()) {
  auto& context = c10::xpu::get_device_context();
  auto& device = c10::xpu::get_raw_device(device_index);
  auto kernel_id = sycl::get_kernel_id<KernelClass>();

  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      context, {device}, {kernel_id});
  sycl::kernel kernel = bundle.get_kernel(kernel_id);
  return kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(
      device);
}

// Overload for functor-based kernels: deduces KernelClass from the function
// object.
template <class KernelClass>
[[nodiscard]] inline size_t getKernelMaxWorkGroupSize(
    const KernelClass& /*kfn*/,
    DeviceIndex device_index = c10::xpu::current_device()) {
  return getKernelMaxWorkGroupSize<KernelClass>(device_index);
}

// Overload for SYCL free-function kernels.
template <auto* KernelFn>
[[nodiscard]] inline size_t getKernelMaxWorkGroupSize(
    DeviceIndex device_index = at::xpu::current_device()) {
  auto& context = c10::xpu::get_device_context();
  auto& device = c10::xpu::get_raw_device(device_index);
  namespace syclex = sycl::ext::oneapi::experimental;
#if SYCL_COMPILER_VERSION < 20260100
  auto bundle =
      syclex::get_kernel_bundle<KernelFn, sycl::bundle_state::executable>(
          context);
  sycl::kernel kernel = bundle.template ext_oneapi_get_kernel<KernelFn>();
  return kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(
      device);
#else
  return syclex::get_kernel_info<
      KernelFn,
      sycl::info::kernel_device_specific::work_group_size>(context, device);
#endif
}

// Returns the maximum number of workitems that are permitted in a subgroup.
[[nodiscard]] inline size_t getDeviceMaxSubGroupSize(
    at::DeviceIndex device = at::xpu::current_device()) {
  const auto* device_prop = at::xpu::getDeviceProperties(device);
  const auto& subgroup_sizes = device_prop->sub_group_sizes;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !subgroup_sizes.empty(),
      "The device subgroup sizes is empty, please check the device status.");
  return *std::max_element(subgroup_sizes.begin(), subgroup_sizes.end());
}

// Returns the minimum number of workitems that are permitted in a subgroup.
[[nodiscard]] inline size_t getDeviceMinSubGroupSize(
    at::DeviceIndex device = at::xpu::current_device()) {
  const auto* device_prop = at::xpu::getDeviceProperties(device);
  const auto& subgroup_sizes = device_prop->sub_group_sizes;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !subgroup_sizes.empty(),
      "The device subgroup sizes is empty, please check the device status.");
  return *std::min_element(subgroup_sizes.begin(), subgroup_sizes.end());
}

// Returns the maximum number of workitems that can be concurrently resident
// on a Xe Core, calculated from the maximum subgroup size, the number of EUs
// per Xe Core, and the number of hardware threads per EU.
[[nodiscard]] inline size_t getDeviceMaxWorkItemsPerXeCore(
    at::DeviceIndex device = at::xpu::current_device()) {
  const auto* device_prop = at::xpu::getDeviceProperties(device);
  return getDeviceMaxSubGroupSize(device) *
      device_prop->gpu_eu_count_per_subslice *
      device_prop->gpu_hw_threads_per_eu;
}

// Returns the number of Xe Cores on the given device.
[[nodiscard]] inline size_t getDeviceXeCoreCount(
    at::DeviceIndex device = at::xpu::current_device()) {
  const auto* device_prop = at::xpu::getDeviceProperties(device);
  return device_prop->gpu_eu_count / device_prop->gpu_eu_count_per_subslice;
}

// Returns the maximum number of workitems that can be concurrently resident
// on the device, based on the maximum subgroup size, the total number of EUs,
// and the number of hardware threads supported per EU.
[[nodiscard]] inline size_t getDeviceMaxWorkItems(
    at::DeviceIndex device = at::xpu::current_device()) {
  const auto* device_prop = at::xpu::getDeviceProperties(device);
  return getDeviceMaxSubGroupSize(device) * device_prop->gpu_eu_count *
      device_prop->gpu_hw_threads_per_eu;
}

// Returns the maximum number of workitems that are permitted in a work-group on
// the given device.
[[nodiscard]] inline size_t getDeviceMaxWorkGroupSize(
    at::DeviceIndex device = at::xpu::current_device()) {
  const auto* device_prop = at::xpu::getDeviceProperties(device);
  return device_prop->max_work_group_size;
}

// Returns the number of hardware threads on the given device.
[[nodiscard]] inline uint32_t getDeviceHWThreads(
    at::DeviceIndex device = at::xpu::current_device()) {
  const auto* device_prop = at::xpu::getDeviceProperties(device);
  return device_prop->gpu_hw_threads_per_eu * device_prop->gpu_eu_count;
}

// Returns the number of EUs per Xe-Core on the given device.
[[nodiscard]] inline uint32_t getDeviceEUCountPerXeCore(
    at::DeviceIndex device = at::xpu::current_device()) {
  const auto* device_prop = at::xpu::getDeviceProperties(device);
  return device_prop->gpu_eu_count_per_subslice;
}

// Returns the share local memory size of the given device.
[[nodiscard]] inline size_t getDeviceLocalMemSize(
    at::DeviceIndex device = at::xpu::current_device()) {
  const auto* device_prop = at::xpu::getDeviceProperties(device);
  return device_prop->local_mem_size;
}

} // namespace at::xpu

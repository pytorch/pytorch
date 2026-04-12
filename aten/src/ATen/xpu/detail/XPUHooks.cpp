#include <ATen/DynamicLibrary.h>
#include <ATen/xpu/PeerToPeerAccess.h>
#include <ATen/xpu/PinnedMemoryAllocator.h>
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUDevice.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <ATen/xpu/detail/LazyLevelZero.h>
#include <ATen/xpu/detail/XPUHooks.h>
#include <ATen/xpu/level_zero_stub/ATenLevelZero.h>
#include <c10/util/Logging.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>

namespace at::xpu::detail {

// Use Level Zero zeMemAllocDevice instead of sycl::aligned_alloc_device.
// The SYCL path triggers DMA-buf host memory shadowing on discrete Intel GPUs
// (Xe2 and later), where the xe kernel driver creates a 1:1 host-side mirror
// for every device allocation. The Level Zero path avoids this.
#ifndef _WIN32
namespace {

void* levelZeroAllocDevice(
    size_t size,
    size_t alignment,
    sycl::device& device,
    sycl::context& context) {
  auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
  auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
  ze_device_mem_alloc_desc_t alloc_desc = {
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void* ptr = nullptr;
  ze_result_t r = at::xpu::detail::lazyLevelZero.zeMemAllocDevice(
      ze_ctx, &alloc_desc, size, alignment, ze_dev, &ptr);
  if (r != ZE_RESULT_SUCCESS || !ptr) {
    return nullptr;
  }
  return ptr;
}

void levelZeroFreeDevice(void* ptr, sycl::context& context) {
  if (!ptr)
    return;
  auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
  at::xpu::detail::lazyLevelZero.zeMemFree(ze_ctx, ptr);
}

bool shouldUseLevelZeroAlloc() {
  const char* env = std::getenv("PYTORCH_XPU_ALLOC_LEVEL_ZERO");
  if (env) {
    return std::string(env) != "0";
  }
  return true;
}

} // namespace
#endif // _WIN32

void XPUHooks::init() const {
  C10_LOG_API_USAGE_ONCE("aten.init.xpu");
  const auto device_count = c10::xpu::device_count_ensure_non_zero();
#ifndef _WIN32
  if (shouldUseLevelZeroAlloc()) {
    c10::xpu::XPUCachingAllocator::setRawDeviceAllocFns(
        levelZeroAllocDevice, levelZeroFreeDevice);
  }
#endif
  c10::xpu::XPUCachingAllocator::init(device_count);
  at::xpu::detail::init_p2p_access_cache(device_count);
}

bool XPUHooks::hasXPU() const {
  return true;
}

std::string XPUHooks::showConfig() const {
  return "XPU backend";
}

int32_t XPUHooks::getGlobalIdxFromDevice(const at::Device& device) const {
  TORCH_CHECK(device.is_xpu(), "Only the XPU device type is expected.");
#if defined(_WIN32) && SYCL_COMPILER_VERSION < 20250000
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "Default context is not supported on XPU by default on Windows for SYCL compiler versions earlier than 2025.0.0. So we can NOT find its global index of the ATen device.");
#else
  return at::xpu::getGlobalIdxFromDevice(device.index());
#endif
}

const Generator& XPUHooks::getDefaultGenerator(DeviceIndex device_index) const {
  return at::xpu::detail::getDefaultXPUGenerator(device_index);
}

Generator XPUHooks::getNewGenerator(DeviceIndex device_index) const {
  return make_generator<at::XPUGeneratorImpl>(device_index);
}

Device XPUHooks::getDeviceFromPtr(void* data) const {
#if defined(_WIN32) && SYCL_COMPILER_VERSION < 20250000
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "Default context is not supported on XPU by default on Windows for SYCL compiler versions earlier than 2025.0.0. So we can NOT find the ATen device of a pointer.");
#else
  return at::xpu::getDeviceFromPtr(data);
#endif
}

/**
 * DEPRECATED: use deviceCount() instead
 */
c10::DeviceIndex XPUHooks::getNumGPUs() const {
  return at::xpu::device_count();
}

/**
 * DEPRECATED: use getCurrentDevice() instead
 */
DeviceIndex XPUHooks::current_device() const {
  return c10::xpu::current_device();
}

void XPUHooks::deviceSynchronize(DeviceIndex device_index) const {
  // Only the SYCL queues we have reserved will be synchronized, see Note
  // [Synchronize Streams on Device].
  c10::xpu::syncStreamsOnDevice(device_index);
}

Allocator* XPUHooks::getPinnedMemoryAllocator() const {
  return at::xpu::getPinnedMemoryAllocator();
}

bool XPUHooks::isPinnedPtr(const void* data) const {
  if (!at::xpu::is_available()) {
    return false;
  }

  return sycl::usm::alloc::host ==
      sycl::get_pointer_type(data, c10::xpu::get_device_context());
}

bool XPUHooks::isAvailable() const {
  return at::xpu::is_available();
}

bool XPUHooks::hasPrimaryContext(DeviceIndex device_index) const {
  // The default context is utilized for each device.
  // So it always returns true if a device is available.
  return isAvailable();
}

DeviceIndex XPUHooks::deviceCount() const {
  return at::xpu::device_count();
}

DeviceIndex XPUHooks::getCurrentDevice() const {
  return at::xpu::current_device();
}

static std::pair<std::unique_ptr<at::DynamicLibrary>, at::xpu::LevelZero*>
load_level_zero() {
  return std::make_pair(nullptr, &at::xpu::detail::lazyLevelZero);
}

const at::xpu::LevelZero& level_zero() {
  static auto handle = load_level_zero();
  return *handle.second;
}

const at::xpu::LevelZero& XPUHooks::level_zero() const {
  return at::xpu::detail::level_zero();
}

REGISTER_XPU_HOOKS(XPUHooks);

} // namespace at::xpu::detail

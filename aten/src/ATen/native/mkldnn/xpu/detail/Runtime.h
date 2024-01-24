#pragma once

#include <ATen/Config.h>

#include <c10/core/Device.h>
#include <c10/xpu/XPUFunctions.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <vector>

namespace at{
namespace native::xpu {
namespace onednn {

// Keep non-static and non-inline
bool set_onednn_verbose(int level);

static inline dnnl::memory xpu_onednn_memory(
    dnnl::memory::desc md,
    dnnl::engine& engine,
    void* ptr) {
  return dnnl::sycl_interop::make_memory(
      md,
      engine,
      dnnl::sycl_interop::memory_kind::usm,
      ptr == nullptr ? DNNL_MEMORY_ALLOCATE : ptr);
}

// GpuEngineManager singleton
struct GpuEngineManager {
  static GpuEngineManager& Instance(); // Singleton

  dnnl::engine& get_engine(const Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == kXPU);
    TORCH_INTERNAL_ASSERT(device.index() < c10::xpu::device_count());
    return *engine_pool[device.index()];
  }

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;

 protected:
  GpuEngineManager() {
    int device_count = (int)c10::xpu::device_count();
    TORCH_INTERNAL_ASSERT(device_count > 0);
    for (int i = 0; i < device_count; i++) {
      engine_pool.push_back(
          std::make_shared<dnnl::engine>(dnnl::sycl_interop::make_engine(
              dpcppGetRawDevice(i), dpcppGetDeviceContext(i))));
    }
  }
  ~GpuEngineManager() {}

 private:
  std::vector<std::shared_ptr<dnnl::engine>> engine_pool;
};

// GpuStreamManager singleton
struct GpuStreamManager {
  static GpuStreamManager& Instance(); // Singleton

  dnnl::stream& get_stream() {
    int device_index = current_device();
    TORCH_INTERNAL_ASSERT(device_index < c10::xpu::device_count());
    int queue_id = getCurrentDPCPPStream(device_index).queue_index();
    if (stream_pool[device_index][queue_id] == nullptr) {
      stream_pool[device_index][queue_id] =
          std::make_shared<dnnl::stream>(dnnl::sycl_interop::make_stream(
              GpuEngineManager::Instance().get_engine({kXPU, device_index}),
              dpcppGetRawQueue(device_index, queue_id)));
    }
    return *(stream_pool[device_index][queue_id].get());
  }

  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;

 protected:
  GpuStreamManager() {
    int deviceCount = c10::xpu::device_count();
    TORCH_INTERNAL_ASSERT(deviceCount > 0);
    stream_pool.clear();
    stream_pool.resize(deviceCount);
    for (DeviceIndex dev = 0; dev < deviceCount; dev++) {
      for (QueueIndex qid = 0; qid < kQueuesPerPool; qid++) {
        stream_pool[dev][qid] = nullptr;
      }
    }
  }
  ~GpuStreamManager() {}

 private:

  // For each device, we have kQueuesPerPool(32) reserved queues.
  std::vector<std::array<std::shared_ptr<dnnl::stream>, kQueuesPerPool>>
      stream_pool;

};

} // namespace onednn
} // namespace native::xpu
} // namespace at

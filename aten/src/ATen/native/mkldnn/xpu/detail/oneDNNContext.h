#pragma once

#include <ATen/Config.h>

#include <c10/core/Device.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <vector>

namespace at::native::onednn {

TORCH_XPU_API dnnl::memory make_onednn_memory(
    dnnl::memory::desc md,
    dnnl::engine& engine,
    void* ptr);

// Keep non-static and non-inline
bool set_onednn_verbose(int level);

// GpuEngineManager singleton
struct TORCH_XPU_API GpuEngineManager {
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
              c10::xpu::get_raw_device(i), c10::xpu::get_device_context()
            )));
    }
  }
  ~GpuEngineManager() {}

 private:
  std::vector<std::shared_ptr<dnnl::engine>> engine_pool;
};

// GpuStreamManager singleton
struct TORCH_XPU_API GpuStreamManager {
  static GpuStreamManager& Instance(); // Singleton

  dnnl::stream get_stream() {
    c10::DeviceIndex device_index = c10::xpu::current_device();
    TORCH_INTERNAL_ASSERT(device_index < c10::xpu::device_count());
    return dnnl::sycl_interop::make_stream(
        GpuEngineManager::Instance().get_engine({c10::kXPU, device_index}),
        c10::xpu::getCurrentXPUStream(device_index).queue());
  }

  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;

 protected:
  GpuStreamManager() {
  }
  ~GpuStreamManager() {}

};

} // namespace at::native::onednn

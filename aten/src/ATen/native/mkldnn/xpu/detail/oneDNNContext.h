#pragma once

#include <ATen/Config.h>

#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>
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

  dnnl::engine& get_engine(
      DeviceIndex device_index = c10::xpu::current_device()) {
    c10::xpu::check_device_index(device_index);
    return *engine_pool[device_index];
  }

  dnnl::engine& get_engine(const Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == kXPU);
    return get_engine(device.index());
  }

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;
  GpuEngineManager(GpuEngineManager&&) = default;
  GpuEngineManager& operator=(GpuEngineManager&&) = default;

 protected:
  GpuEngineManager();
  ~GpuEngineManager() = default;

 private:
  std::vector<std::shared_ptr<dnnl::engine>> engine_pool;
};

// GpuStreamManager singleton
struct TORCH_XPU_API GpuStreamManager {
  static GpuStreamManager& Instance(); // Singleton

  dnnl::stream& get_stream(
      DeviceIndex device_index = c10::xpu::current_device()) {
    auto stream = c10::xpu::getCurrentXPUStream(device_index);
    auto priority = stream.priority();
    if (stream_pool[device_index][priority].find(stream) ==
        stream_pool[device_index][priority].end()) {
      stream_pool[device_index][priority][stream] =
          std::make_shared<dnnl::stream>(dnnl::sycl_interop::make_stream(
              GpuEngineManager::Instance().get_engine(device_index),
              stream.queue()));
    }
    return *stream_pool[device_index][priority][stream];
  }

  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;
  GpuStreamManager(GpuStreamManager&&) = default;
  GpuStreamManager& operator=(GpuStreamManager&&) = default;

 protected:
  GpuStreamManager() {
    c10::DeviceIndex device_count = c10::xpu::device_count_ensure_non_zero();
    stream_pool.resize(device_count);
  }
  ~GpuStreamManager() = default;

 private:
  using stream_hash_map =
      ska::flat_hash_map<c10::xpu::XPUStream, std::shared_ptr<dnnl::stream>>;
  std::vector<
      std::array<stream_hash_map, c10::xpu::max_compile_time_stream_priorities>>
      stream_pool;
};

} // namespace at::native::onednn

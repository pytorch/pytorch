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

  dnnl::engine& get_engine(const Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == kXPU);
    TORCH_INTERNAL_ASSERT(device.index() < c10::xpu::device_count());
    return *engine_pool[device.index()];
  }

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;

 protected:
  GpuEngineManager() {
    c10::DeviceIndex device_count = c10::xpu::device_count();
    TORCH_INTERNAL_ASSERT(device_count > 0);
    for (const auto i : c10::irange(device_count)) {
      engine_pool.push_back(
          std::make_shared<dnnl::engine>(dnnl::sycl_interop::make_engine(
              c10::xpu::get_raw_device(i), c10::xpu::get_device_context())));
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
    auto stream = c10::xpu::getCurrentXPUStream();
    auto priority = stream.priority();
    auto device_index = stream.device_index();
    if (stream_pool[device_index][priority].find(stream) ==
        stream_pool[device_index][priority].end()) {
      stream_pool[device_index][priority][stream] =
          std::make_shared<dnnl::stream>(dnnl::sycl_interop::make_stream(
              GpuEngineManager::Instance().get_engine(
                  {c10::kXPU, device_index}),
              stream.queue()));
    }
    return *stream_pool[device_index][priority][stream];
  }

  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;

 protected:
  GpuStreamManager() {
    c10::DeviceIndex device_count = c10::xpu::device_count();
    TORCH_INTERNAL_ASSERT(device_count > 0);
    stream_pool.resize(device_count);
  }
  ~GpuStreamManager() {}

 private:
  using stream_hash_map =
      ska::flat_hash_map<c10::xpu::XPUStream, std::shared_ptr<dnnl::stream>>;
  std::vector<
      std::array<stream_hash_map, c10::xpu::max_compile_time_stream_priorities>>
      stream_pool;
};

} // namespace at::native::onednn

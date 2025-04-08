#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>

/* *
 * Do NOT put any kernels or call any device binaries here!
 * Only maintain oneDNN runtime states in this file.
 * */
namespace at::native::onednn {

using namespace dnnl;

static inline void* dnnl_alloc(
    size_t size,
    size_t /*alignment*/,
    const void* /*dev*/,
    const void* /*context*/) {
  return c10::xpu::XPUCachingAllocator::raw_alloc(size);
}

static inline void dnnl_delete(
    void* buf,
    const void* /*dev*/,
    const void* /*context*/,
    void* /*event*/) {
  return c10::xpu::XPUCachingAllocator::raw_delete(buf);
}

GpuEngineManager::GpuEngineManager() {
  c10::DeviceIndex device_count = c10::xpu::device_count();
  TORCH_INTERNAL_ASSERT(device_count > 0);
  for (const auto i : c10::irange(device_count)) {
    static dnnl::graph::allocator alloc =
        dnnl::graph::sycl_interop::make_allocator(dnnl_alloc, dnnl_delete);
    engine_pool.push_back(std::make_shared<dnnl::engine>(
        dnnl::graph::sycl_interop::make_engine_with_allocator(
            c10::xpu::get_raw_device(i),
            c10::xpu::get_device_context(),
            alloc)));
  }
}

GpuEngineManager& GpuEngineManager::Instance() {
  static GpuEngineManager myInstance;
  return myInstance;
}

GpuStreamManager& GpuStreamManager::Instance() {
  static thread_local GpuStreamManager myInstance;
  return myInstance;
}

bool set_onednn_verbose(int level) {
  dnnl::status rs = dnnl::set_verbose(level);
  return rs == dnnl::status::success;
}

} // namespace at::native::onednn

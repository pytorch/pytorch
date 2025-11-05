#include <ATen/xpu/CachingHostAllocator.h>

namespace at::xpu {
namespace {

constexpr size_t kHostAlignment = 512;

using Block = HostBlock<XPUStream>;

struct XPUCachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<XPUStream, XPUEvent> {
  /* These following functions are runtime-related. */
  void allocate_host_memory(size_t size, void** ptr) override {
    *ptr = sycl::aligned_alloc_host(
        kHostAlignment, size, c10::xpu::get_device_context());
  }

  void free_block(Block* block) override {
    sycl::free(block->ptr_, c10::xpu::get_device_context());
  }

  void record_stream(
      std::optional<std::vector<XPUEvent>>& events,
      XPUStream stream) override {
    XPUEvent event;
    event.record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(XPUEvent& event) override {
    return event.query();
  }

  bool pinned_use_background_threads() override {
    // Using background threads for XPU causes a hang on Windows during program
    // exit. Will be enabled once the issue is resolved.
    return false;
  }
};

DECLARE_HOST_ALLOCATOR(
    XPUCachingHostAllocator,
    XPUCachingHostAllocatorImpl,
    raw_local_deleter,
    caching_host_allocator)

REGISTER_HOST_ALLOCATOR(at::kXPU, &caching_host_allocator);

} // anonymous namespace
} // namespace at::xpu

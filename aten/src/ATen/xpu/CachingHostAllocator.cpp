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
};

void raw_local_deleter(void* ptr);

struct XPUCachingHostAllocator final
    : public CachingHostAllocatorInterface<XPUCachingHostAllocatorImpl> {
  at::DataPtr allocate(size_t size) override {
    auto ptr_and_ctx = impl_->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &raw_local_deleter,
        at::DeviceType::CPU};
  }
};

static XPUCachingHostAllocator caching_host_allocator;

static inline XPUCachingHostAllocator& getXPUCachingHostAllocator() {
  return caching_host_allocator;
}

void raw_local_deleter(void* ptr) {
  getXPUCachingHostAllocator().free(ptr);
}

} // anonymous namespace

bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::xpu::XPUStream stream) {
  return getXPUCachingHostAllocator().record_event(ptr, ctx, stream);
}

void CachingHostAllocator_emptyCache() {
  getXPUCachingHostAllocator().empty_cache();
}

at::Allocator* getCachingHostAllocator() {
  return &getXPUCachingHostAllocator();
}

} // namespace at::xpu

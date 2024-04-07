#include <ATen/xpu/CachingHostAllocator.h>

namespace at::xpu {
namespace {

constexpr size_t kHostAlignment = 512;

using Block = HostBlock<XPUStream>;
using Comparator = ComparatorSize<Block>;
using AllocatorImplInterface =
    CachingHostAllocatorImplInterface<XPUStream, XPUEvent, Block, Comparator>;

struct XPUHostAllocatorImpl : public AllocatorImplInterface {
  void allocate_host_memory(size_t size, void** ptr) override {
    *ptr = sycl::aligned_alloc_host(
        kHostAlignment, size, c10::xpu::get_device_context());
  }

  void free_block(Block* block) override {
    sycl::free(block->ptr_, c10::xpu::get_device_context());
  }

  void record_stream(
      c10::optional<std::vector<XPUEvent>>& events,
      XPUStream stream) override {
    XPUEvent event;
    event.record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(XPUEvent& event) override {
    return event.query();
  }
};

} // anonymous namespace

void raw_local_deleter(void* ptr);

struct XPUHostAllocator final
    : public HostAllocatorInterface<XPUHostAllocatorImpl> {
  at::DataPtr allocate(size_t size) override {
    auto ptr_and_ctx = impl_->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &raw_local_deleter,
        at::DeviceType::CPU};
  }
};

static XPUHostAllocator host_allocator;

void raw_local_deleter(void* ptr) {
  host_allocator.free(ptr);
}

static inline XPUHostAllocator& getXPUHostAllocator() {
  return host_allocator;
}

bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::xpu::XPUStream stream) {
  return getXPUHostAllocator().record_event(ptr, ctx, stream);
}

void CachingHostAllocator_emptyCache() {
  getXPUHostAllocator().empty_cache();
}

at::Allocator* getCachingHostAllocator() {
  return &host_allocator;
}

} // namespace at::xpu

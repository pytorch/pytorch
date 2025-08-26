#include <ATen/cuda/CachingHostAllocator.h>

#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/thread_pool.h>
#include <c10/cuda/CUDAAllocatorConfig.h>

#include <cuda_runtime_api.h>
#include <future>
#include <unordered_map>

namespace at::cuda {
namespace {

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<
      at::cuda::CUDAEvent,
      std::function<void(at::cuda::CUDAEvent*)>>;
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](at::cuda::CUDAEvent* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<at::cuda::CUDAEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(
        std::make_unique<at::cuda::CUDAEvent>(cudaEventDisableTiming).release(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<at::cuda::CUDAEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

using Block = HostBlock<CUDAStream>;

struct CUDACachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<CUDAStream, EventPool::Event> {
 private:
  std::unordered_map<void*, bool> use_host_register;

  void allocate_host_memory(size_t size, void** ptr) override {
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    // This can be a large performance hit if we cross NUMA nodes by allocating
    // and pinning memory on one side of the NUMA node and then using it on the
    // other side. Thankfully, we use one process per GPU, so we don't run into
    // this issue.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::cuda::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
    }

    auto start = std::chrono::steady_clock::now();
    bool use_register = c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_use_cuda_host_register();
    if (use_register) {
      allocWithCudaHostRegister(ptr, size);
    } else {
      // Use cudaHostAlloc for allocating pinned memory (global lock in driver)
      C10_CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on cudaHostAlloc/hostRegister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(use_host_register.count(*ptr) == 0);
      use_host_register[*ptr] = use_register;
      stats_.host_alloc_time.increase(duration.count());
    }
  }

  void free_block(Block* block) override {
    auto start = std::chrono::steady_clock::now();
    // Users may change the allocator config at will. torch unit tests do this.
    // However, allocations using cudaHostRegister should use corresonding
    // cudaHostUnregister and similarly for cudaHostAlloc / cudaFreeHost.
    void* ptr = block->ptr_;
    bool use_register = false;
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(use_host_register.count(ptr) == 1);
      use_register = use_host_register[ptr];
    }
    if (use_register) {
      AT_CUDA_CHECK(cudaHostUnregister(ptr));
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      std::free(ptr);
    } else {
      AT_CUDA_CHECK(cudaFreeHost(ptr));
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on cudaFreeHost/hostUnregister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      use_host_register.erase(ptr);
      stats_.host_free_time.increase(duration.count());
    }
  }

  void record_stream(
      std::optional<std::vector<EventPool::Event>>& events,
      CUDAStream stream) override {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(EventPool::Event& event) override {
    cudaError_t err = cudaEventQuery(*event);
    if (err == cudaErrorNotReady) {
      (void)cudaGetLastError(); // clear CUDA error
      return false;
    } else if (err != cudaSuccess) {
      C10_CUDA_CHECK(err);
    }
    return true;
  }

  bool pinned_use_background_threads() override {
    return c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
        pinned_use_background_threads();
  }

  EventPool::Event create_event_internal(DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  TaskThreadPool* getThreadPool() {
    static TaskThreadPool* pool = new TaskThreadPool(
        static_cast<int>(c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
            pinned_max_register_threads()));
    return pool;
  }

  void mapPagesForRegister(
      const void* ptr,
      size_t size,
      size_t i,
      size_t numThreads,
      size_t pageSize) {
    uintptr_t start = (uintptr_t)ptr + (size * i / numThreads);
    uintptr_t end = (uintptr_t)start + (size / numThreads);
    if (i == (numThreads - 1)) {
      end = (uintptr_t)ptr + size;
    }

    // pre-fault/map the pages by setting the first byte of the page
    uintptr_t alignedStart =
        (((uintptr_t)start + pageSize - 1) & ~(pageSize - 1));
    for (uintptr_t p = alignedStart; p < ((uintptr_t)end); p += pageSize) {
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      memset((void*)p, 0, 1);
    }
  }

  void allocWithCudaHostRegister(void** ptr, size_t roundSize) {
    // Here we do regular allocation, pre-fault/map the pages, and then do
    // cudaHostRegister with GPU mapping flags to lock the pages, so we
    // can minimize the cost for the cuda global lock.
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    *ptr = std::malloc(roundSize);

    // Parallelize the mapping/registering of pages to reduce wall time
    size_t pageSize = (1 << 12); // 4kB pages
    size_t numMapThreads = c10::cuda::CUDACachingAllocator::
        CUDAAllocatorConfig::pinned_num_register_threads();
    if ((numMapThreads > 1) && (roundSize >= (pageSize * numMapThreads))) {
      // parallelize the mapping of pages with a threadpool
      auto* pool = getThreadPool();
      std::vector<std::promise<void>> promises;
      std::vector<std::future<void>> futures;
      promises.reserve(numMapThreads);
      futures.reserve(numMapThreads);

      for (size_t i = 0; i < numMapThreads; i++) {
        promises.emplace_back();
        futures.push_back(promises[i].get_future());
        auto task = [this,
                     i,
                     ptr,
                     roundSize,
                     numMapThreads,
                     pageSize,
                     &promises]() mutable {
          mapPagesForRegister(
              *ptr,
              roundSize,
              i, // thread task-id
              numMapThreads,
              pageSize);
          // set the promise when mapping pages are done
          promises[i].set_value();
        };
        pool->run(task);
      }
      for (auto& future : futures) {
        future.wait();
      }
    } else {
      // Map pages in the same thread
      mapPagesForRegister(*ptr, roundSize, 0, 1, pageSize);
    }

    // Register the mapped pages using cudaHostRegister
    AT_CUDA_CHECK(
        cudaHostRegister(*ptr, roundSize, cudaHostRegisterDefault));
  }
};

DECLARE_HOST_ALLOCATOR(
    CUDACachingHostAllocator,
    CUDACachingHostAllocatorImpl,
    raw_local_deleter,
    caching_host_allocator)

REGISTER_HOST_ALLOCATOR(at::kCUDA, &caching_host_allocator)

} // anonymous namespace
} // namespace at::cuda

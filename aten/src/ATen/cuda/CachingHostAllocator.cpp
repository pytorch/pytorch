#include <ATen/cuda/CachingHostAllocator.h>

#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/thread_pool.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/util/hash.h>

#include <cuda_runtime_api.h>
#include <future>

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
  ska::flat_hash_map<void*, bool> use_host_register;
public:
  bool record_event(void* ptr, void* ctx, c10::Stream s) override {
    if (!CachingHostAllocatorImpl<CUDAStream, EventPool::Event>::record_event(ptr, ctx, s)) {
      return false;
    }

    // Inspect the preceding captured node on stream s. If it's a
    // memcpy from pinned host memory that reads from this block, hash
    // the copied section and record it to detect accidental mutations
    // during capture. The purpose is to identify cases where the CPU
    // writes to host pinned memory after the GPU is scheduled to read
    // from it during cuda graph replay. I am calling this a
    // CPU-Write-After-GPU-Read hazard. Such a situation is almost
    // certainly a bug, bcause it doesn't match the semantics of eager
    // launch. That check happens in end_allocate_to_pool.

    Block *block = nullptr;
    if (block_exists(ctx)) {
      block = reinterpret_cast<Block*>(ctx);
    } else {
      block = get_block_from_ptr(ptr);
    }

    assert(block != nullptr);

    CUDAStream stream(CUDAStream::UNCHECKED, s);

    std::optional<std::tuple<const cudaGraphNode_t*, size_t>> terminals_opt = c10::cuda::streamGetTerminalNodes(stream);

    if (!terminals_opt.has_value()) {
      // no stream capture is happening right now
      return true;
    }

    auto&& [terminals, num_terminals] = *terminals_opt;

    if (num_terminals == 0) {
      TORCH_WARN(
          "CachingHostAllocator: no nodes precede record_event() during stream capture; "
          "this is probably a bug.");
      return true;
    }

    if (num_terminals > 1) {
      TORCH_WARN(
          "CachingHostAllocator: multiple preceding nodes during capture; "
          "skipping host read verification.");
      return true;
    }

    cudaGraphNode_t node = terminals[0];
    cudaGraphNodeType type{cudaGraphNodeTypeCount};
    C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeMemcpy) {
      TORCH_WARN(
          "CachingHostAllocator: preceding captured node is not a memcpy; "
          "can only check for CPU-write-after-GPU-read errors for memcpy nodes.");
      return true;
    }

    cudaMemcpy3DParms params{};
    C10_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &params));

    // Only handle Host->Device copies (source is host memory)
    if (params.kind != cudaMemcpyHostToDevice && params.kind != cudaMemcpyDefault) {
      // don't warn, since Device->Host copy is a valid node to precede record_event()
      return true;
    }

    cudaPointerAttributes attr{};
    cudaError_t attr_err = cudaPointerGetAttributes(&attr, params.srcPtr.ptr);
    // TODO: Think about what to do if the memory type is
    // cudaMemoryTypeUnregistered, which I believe refers to pageable
    // host memory. (Meanwhile, cudaMemoryTypeHost represents pinned
    // host memory.)
    if (attr_err != cudaSuccess || attr.type != cudaMemoryTypeHost) {
      // Clear error if any and return (not pinned host memory)
      (void)cudaGetLastError();
      TORCH_WARN("CachingHostAllocator: source of memcpy is not pinned host memory");
      return true;
    }

    // Compute the start pointer and a conservative contiguous span size.
    // For typical 1D host->device copies, height=depth=1 and srcPos.y/z=0.
    if (params.extent.height > 1 || params.extent.depth > 1) {
      TORCH_WARN("CachingHostAllocator: non-1D memcopies are not supported for CPU-write-after-GPU-read detection. File an issue if this is important to you.");
      return true;
    }
    char* start_ptr = static_cast<char*>(params.srcPtr.ptr) + params.srcPos.x;
    size_t n = params.extent.width;

    if (n == 0) {
      TORCH_WARN("CachingHostAllocator: cudaMemcpyAsync node is copying 0 bytes");
      return true;
    }

    // Verify the memcpy source buffer is within this block's bounds
    char *block_start, *block_end;
    {
      std::lock_guard<std::mutex> lg(block->mutex_);
      block_start = static_cast<char*>(block->ptr_);
      block_end = block_start + block->size_;
    }
    char* src_end = start_ptr + n;
    if (start_ptr < block_start || src_end > block_end) {
      TORCH_WARN("CachingHostAllocator: cudaMemcpyAsync node is copying memory outside this block's memory region, which shouldn't be possible.");
      return true;
    }

    // Hash the source buffer and record the tuple (ptr, size, hash)
    c10::sha1 hasher(start_ptr, n);
    std::string hash_string = hasher.str();
    {
      std::lock_guard<std::mutex> lg(block->mutex_);
      block->sections_read_under_stream_capture_.emplace_back(
          static_cast<void*>(start_ptr), n, std::move(hash_string));
    }
    return true;
  }

  // See comment in the header: we need to validate that any pinned host
  // memory regions read by captured memcpy nodes have not been modified during
  // capture. Perform the validation at the end of allocation-to-pool.
  void end_allocate_to_pool(c10::MempoolId_t pool_id) {
    // Look up the private pool for this capture.
    {
      std::shared_lock<std::shared_mutex> lg(instance_mutex_);
      auto it = graph_pools_.find(pool_id);
      if (it != graph_pools_.end()) {
        auto& pool = it->second->blocks;
        // Lock blocks set while we iterate
        std::lock_guard<std::mutex> gb(pool.blocks_mutex_);
        for (auto* block : pool.blocks_) {
          // Copy sections under block lock and clear them to avoid rechecking
          std::vector<std::tuple<void*, size_t, std::string>> sections;
          {
            std::lock_guard<std::mutex> bl(block->mutex_);
            sections = block->sections_read_under_stream_capture_;
            block->sections_read_under_stream_capture_.clear();
          }
          for (const auto& tup : sections) {
            void* start = nullptr;
            size_t len = 0;
            const std::string* expected_hash = nullptr;
            // tie cannot bind to temporary from get<>, extract manually
            start = std::get<0>(tup);
            len = std::get<1>(tup);
            expected_hash = &std::get<2>(tup);
            c10::sha1 hasher(start, len);
            std::string actual = hasher.str();
            TORCH_CHECK(
                actual == *expected_hash,
                "CUDA graph replay will not match eager execution: a pinned host "
                "memory buffer read by a memcpy during capture was modified after "
                "it was recorded. To fix, never write to pinned host memory after "
                "stream capture records a memcpy that reads from it.");
          }
        }
      }
    }

    // Defer to base implementation to end allocation to this pool.
    CachingHostAllocatorImpl<CUDAStream, EventPool::Event>::end_allocate_to_pool(
        pool_id);
  }

 private:

  void allocate_host_memory(size_t size, void** ptr) override {
    // try allocating from reserve segment first before calling into expensive APIs
    if (get_reserve_segment().initialized()) {
      *ptr = get_reserve_segment().allocate(size);
      if (*ptr != nullptr) {
        return;
      }
    }
    allocate_host_memory_slowpath(size, ptr);
  }

  void allocate_host_memory_slowpath(size_t size, void** ptr) {
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
      at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
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
    // We never free blocks from the reserve segment
    if (get_reserve_segment().initialized()) {
      // Check if the block is from the reserve segment
      if (get_reserve_segment().owns(block->ptr_)) {
        return;
      }
    }

    free_block_slowpath(block);
  }

  void free_block_slowpath(Block* block) {
    auto start = std::chrono::steady_clock::now();
    // Users may change the allocator config at will. torch unit tests do this.
    // However, allocations using cudaHostRegister should use corresponding
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
    // It is rare, but query_event() can be called during stream
    // capture. This happens when stream capture is immediately
    // preceded by allocating to this pool via
    // _use_cuda_memory_pool_manager. Since capture_begin() is
    // preceded by torch.cuda.synchronize() in torch/cuda/graphs.py,
    // we can be certain that cudaEventQuery always returns
    // cudaSuccess in this rare situation.
    at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
    cudaError_t err = cudaEventQuery(*event);
    if (err == cudaErrorNotReady) {
      (void)cudaGetLastError(); // clear CUDA error
      return false;
    } else if (err != cudaSuccess) {
      C10_CUDA_CHECK(err);
    }
    return true;
  }

  EventPool::Event create_event_internal(DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  PinnedReserveSegment& get_reserve_segment() {
    static auto reserve_segment = [&]() {
      if (c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_reserve_segment_size_mb() > 0) {
        void *ptr;
        size_t sz = c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_reserve_segment_size_mb() * 1024 * 1024;
        allocate_host_memory_slowpath(sz, &ptr);
        return PinnedReserveSegment(ptr, sz);
      } else {
        return PinnedReserveSegment();
      }
    } ();
    return reserve_segment;
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
    uintptr_t end = start + (size / numThreads);
    if (i == (numThreads - 1)) {
      end = (uintptr_t)ptr + size;
    }

    // pre-fault/map the pages by setting the first byte of the page
    uintptr_t alignedStart =
        ((start + pageSize - 1) & ~(pageSize - 1));
    for (uintptr_t p = alignedStart; p < (end); p += pageSize) {
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
    at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
    AT_CUDA_CHECK(
        cudaHostRegister(*ptr, roundSize, cudaHostRegisterDefault));
  }

  CUDAStream get_current_stream() const override {
    return at::cuda::getCurrentCUDAStream();
  }

  bool stream_is_capturing(CUDAStream s) const override {
    cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
    C10_CUDA_CHECK(cudaStreamIsCapturing(s, &status));
    return status != cudaStreamCaptureStatusNone;
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

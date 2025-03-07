#include <ATen/cuda/CachingHostAllocator.h>

#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/thread_pool.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <c10/util/flat_hash_map.h>

#include <cuda_runtime_api.h>
#include <future>
#include <iostream>
#include <unordered_set>
#include <vector>


namespace at::cuda {

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

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

    auto start = std::chrono::system_clock::now();
    if (c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
            pinned_use_cuda_host_register()) {
      allocWithCudaHostRegister(ptr, size);
    } else {
      // Use cudaHostAlloc for allocating pinned memory (global lock in driver)
      C10_CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on cudaHostAlloc/hostRegister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      stats_.host_alloc_time.increase(duration.count());
    }
  }

  void free_block(Block* block) override {
    auto start = std::chrono::system_clock::now();
    if (c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
            pinned_use_cuda_host_register()) {
      void* ptr = block->ptr_;
      AT_CUDA_CHECK(cudaHostUnregister(ptr));
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      std::free(ptr);
    } else {
      AT_CUDA_CHECK(cudaFreeHost(block->ptr_));
    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on cudaFreeHost/hostUnregister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
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

  void registerPages(const void* ptr, size_t size) {
    AT_CUDA_CHECK(
                  cudaHostRegister((void*)ptr, (size_t)size, cudaHostRegisterDefault));

    // If host and device pointer don't match, give a warning and exit
    void* devptr = nullptr;
    AT_CUDA_CHECK(cudaHostGetDevicePointer(&devptr, (void*)ptr, 0));
    TORCH_CHECK(
        (void*)devptr == (void*)ptr,
        "Host and device pointer dont match with cudaHostRegister. "
        "Please dont use this feature by setting "
        "PYTORCH_CUDA_ALLOC_CONF=use_cuda_host_register:False (default)",
        "");
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
    registerPages(*ptr, roundSize);
  }
};

void raw_local_deleter(void* ptr);

struct CUDACachingHostAllocator final
    : public CachingHostAllocatorPimpl<CUDACachingHostAllocatorImpl, CUDAStream> {
  at::DataPtr allocate(size_t size) override {
    auto ptr_and_ctx = impl_->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &raw_local_deleter,
        at::DeviceType::CPU};
  }
};

struct GraphBlock {
  GraphBlock(void* ptr_, std::size_t size_) : ptr(ptr_), size(size_) {}

  void *ptr;
  std::size_t size;
  std::vector<cudaGraphNode_t> using_nodes;
  bool allocated_{true};
};

struct PrivatePool {
  // unlike CUDACachingHostAllocator,
  // CUDAStreamCapturableCachingHostAllocator uses only a single mutex
  // for the entirety of a memory pool. There are a few reasons:

  // 1. simplicity. Stream capture rarely spans multiple threads in
  // pytorch. The main situation in which this happens (that I know of) is when
  // 2. CUDAStreamCapturableCachingHostAllocator is used only during stream capture,
  // which is allowed to be "slow".
  // 3. The only expensive API call we do is cudaHostAlloc(). Events are no longer used.
  alignas(64) std::mutex mutex;
  int use_count{1};
  ska::flat_hash_set<GraphBlock*> blocks;

  // We keep free list as a vector of free lists, one for each power of two
  // size. This allows us to quickly find a free block of the right size.
  // We use deque to store per size free list and guard the list with its own
  // mutex.
  std::vector<FreeBlockList<GraphBlock>> free_list_ = std::vector<FreeBlockList<GraphBlock>>(MAX_SIZE_INDEX);
};

cudaGraphNode_t insert_empty_node(GraphBlock* context, CUDAStream stream) {
    cudaStreamCaptureStatus status{};
    cudaGraph_t currently_capturing_graph{};
    const cudaGraphNode_t* dependencies{};
    size_t num_dependencies = 0;
    AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream,
                                               &status,
                                               nullptr,
                                               &currently_capturing_graph,
                                               &dependencies,
                                               &num_dependencies));
    TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatusActive);

    cudaGraphNode_t new_node{};
    AT_CUDA_CHECK(cudaGraphAddEmptyNode(
                                       &new_node,
                                       currently_capturing_graph,
                                       dependencies,
                                       num_dependencies));

    AT_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
                                                      stream, &new_node, 1, cudaStreamSetCaptureDependencies));

    if (context != nullptr) {
      context->using_nodes.push_back(new_node);
    }

    return new_node;
}

bool path_exists(cudaGraph_t graph, cudaGraphNode_t source,
                 cudaGraphNode_t destination) {
    // Use an unordered_set to track visited nodes.
    std::unordered_set<cudaGraphNode_t> visited;

    // Define a recursive lambda to perform DFS.
    std::function<bool(cudaGraphNode_t)> dfs = [&](cudaGraphNode_t current) -> bool {
        if (current == destination) {
            return true;
        }
        visited.insert(current);

        // First, get the number of dependent nodes.
        size_t numDependent = 0;
        AT_CUDA_CHECK(cudaGraphNodeGetDependentNodes(current, nullptr, &numDependent));
        // Allocate a vector to hold the dependent nodes.
        std::vector<cudaGraphNode_t> dependents(numDependent);
        AT_CUDA_CHECK(cudaGraphNodeGetDependentNodes(current, dependents.data(), &numDependent));
        // Recursively search each dependent node.
        for (const auto& next : dependents) {
            if (visited.find(next) == visited.end()) {
                if (dfs(next)) {
                    return true;
                }
            }
        }
        return false;
    };

    return dfs(source);
}

void stream_capturable_raw_local_deleter(void* ptr);

/**
 * CUDAStreamCapturableCachingHostAllocator is used only during stream
 * capture.  It is accessed from CUDAGraph.cpp via the functions in
 * CachingHostAllocator.h.  In particular,
 * getCUDACachingHostAllocator() will select the singleton
 * CUDAStreamCapturableCachingHostAllocator whenever the current
 * stream is doing stream capture.
 *
 * Unlike CUDACachingHostAllocator,
 * CUDAStreamCapturableCachingHostAllocator cannot query events
 * associated with usages to detect when a host allocation is free to
 * be reused. This is because cuda events will never actually be
 * recorded during stream capture. They will be recorded only during
 * graph replay. Therefore, during stream capture to a graph, a naive
 * implementation based on the original design would simply fail to
 * reuse any allocations whatsoever, within a cuda graph, which is not
 * acceptable for a caching allocator.
 *
 * This implementation uses a simple algorithm to decide whether a
 * particular host allocation block can be recycled. At every instance
 * of record_event(), it will insert an empty node into the currently
 * capturing stream. If (1) the reference count of a cudaHostAlloc()
 * created block has gone to 0 (tracked via the `allocated` field in
 * HostBlock), and (2) there is a path from every empty node created
 * by a call to record_event() to the current node in stream capture,
 * then this block can be reused, and is therefore moved to the free
 * list.
 *
 * TODO: Consider how to make this work in the case where a memory
 * pool is shared across several cuda graphs. Especially when we have
 * external events, which break the assumption in pytorch that cuda
 * graphs are "atomic" units of execution. We probably need to treat
 * the insertion of an external event into the graph as a possible
 * "usage" with an indeterminate end point, which isn't great because
 * it is very pessimizing.
 *
 * External events PR: https://github.com/pytorch/pytorch/pull/146145
 */
struct CUDAStreamCapturableCachingHostAllocator final
  : public CachingHostAllocatorInterface<CUDAStream> {

  CUDAStreamCapturableCachingHostAllocator() = default;

  void add_to_block_list(PrivatePool& cuda_mem_pool, GraphBlock *block) {
    std::lock_guard<std::mutex> g(cuda_mem_pool.mutex);
    cuda_mem_pool.blocks.insert(block);
  }

  at::DataPtr allocate(size_t size) override {
    void* host_ptr = nullptr;
    CUDAStream stream = getCurrentCUDAStream();

    std::lock_guard<std::mutex> g(m);

    // in a multi-threaded context, this wouldn't work. captures_underway_ could contain multiple instances of the smae mempool_id...
    for (auto &&[mempool_id, filter]: captures_underway_) {
      if (filter(stream)) {
        PrivatePool& cuda_mem_pool = *cuda_mem_pools_.at(mempool_id);

        size_t round_size = c10::llvm::PowerOf2Ceil(size);

        // TODO: Consider doing this only if there is no round_size block on the free list.
        // We don't want allocation to be O(N), where N is the number of active blocks.
        free_finished_allocations(cuda_mem_pool);

        // First, try to allocate from the free list
        auto* block = get_free_block(cuda_mem_pool, round_size);
        if (block) {
          // do I want to pass raw_local_deleter?
          add_to_block_list(cuda_mem_pool, block);
          return at::DataPtr(block->ptr, block, stream_capturable_raw_local_deleter, DeviceType::CPU);
        } else {
          at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
          // std::cout << "GALVEZ:cudaHostAlloc()" << std::endl;
          AT_CUDA_CHECK(cudaHostAlloc(&host_ptr, round_size, cudaHostAllocDefault));
          // Need to make sure to free this...
          block = new GraphBlock(host_ptr, round_size);
          add_to_block_list(cuda_mem_pool, block);
          return at::DataPtr(host_ptr, block, stream_capturable_raw_local_deleter, DeviceType::CPU);
        }
      }
    }

    TORCH_INTERNAL_ASSERT(false, "CUDAStreamCapturableCachingHostAllocator::allocate() is expected to be called only within a capturing context");
  }

  inline size_t size_index(size_t size) {
    return c10::llvm::Log2_64_Ceil(size);
  }

  GraphBlock* get_free_block(PrivatePool& cuda_mem_pool, size_t size) {
    auto index = size_index(size);
    std::lock_guard<std::mutex> g(cuda_mem_pool.mutex);
    FreeBlockList<GraphBlock>& free_blocks = cuda_mem_pool.free_list_[index];
    if (free_blocks.list_.empty()) {
      GraphBlock* block = free_blocks.list_.back();
      free_blocks.list_.pop_back();
      TORCH_INTERNAL_ASSERT(!block->allocated_);
      block->allocated_ = true;
      return block;
    }
    return nullptr;
  }

  void free_finished_allocations(PrivatePool& cuda_mem_pool) {
    std::lock_guard<std::mutex> g(cuda_mem_pool.mutex);

    // WARNING: This is O(N), where N is number of live blocks.
    for (GraphBlock* block: cuda_mem_pool.blocks) {
      if (block->allocated_) {
        continue;
      }
      // is it okay to insert into empty nodes into any random stream?
      // This a mutation, so it's note great... Though it *shouldn't*
      // have any user-perceptible side effects.
      CUDAStream stream = getCurrentCUDAStream();
      cudaGraphNode_t destination_node = insert_empty_node(nullptr, stream);
      bool all_usages_done = true;

      cudaStreamCaptureStatus status{};
      cudaGraph_t currently_capturing_graph{};
      const cudaGraphNode_t* dependencies{};
      size_t num_dependencies = 0;
      AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream,
                                             &status,
                                             nullptr,
                                             &currently_capturing_graph,
                                             &dependencies,
                                             &num_dependencies));
      TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatusActive);


      // there is a problem with this approach! What if we capture to
      // two graphs with the same memory pool? We have no way to detect
      // that "output allocations" of a graph are done being used...
      for (cudaGraphNode_t source_node: block->using_nodes) {
        if (!path_exists(currently_capturing_graph, source_node, destination_node)) {
          all_usages_done = false;
        }
      }

      if (all_usages_done) {
        block->using_nodes.clear();
        // TODO: Move block to free list.
        auto index = size_index(block->size);
        FreeBlockList<GraphBlock>& free_blocks = cuda_mem_pool.free_list_[index];
        free_blocks.list_.push_front(block);
      }
    }
  }

  void free(void* ctx) override {
    auto block = (GraphBlock*)ctx;
    block->allocated_ = false;
    // Do nothing: A cuda graph's allocations will never be freed
  }

  bool record_event(void* ptr, void* ctx, CUDAStream stream) override {
    insert_empty_node((GraphBlock*)ctx, stream);
    return true;
  }

  void empty_cache() override {
    // We cannot release any memory if it has been allocated within a cuda
    // graph, until that cuda graph gets destroyed. So this is a no-op
    // operation.
  }

  void copy_data(void* dest [[maybe_unused]], const void* src [[maybe_unused]], std::size_t count [[maybe_unused]]) const override {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for copy_data");
  }

  void begin_allocate_to_pool(std::pair<unsigned long long, unsigned long long> pool_id, std::function<bool(CUDAStream)> filter) {
    std::lock_guard<std::mutex> g(m);
    if (!cuda_mem_pools_.count(pool_id)) {
      cuda_mem_pools_.emplace(pool_id, std::make_unique<PrivatePool>());
    } else {
      PrivatePool& pool = *cuda_mem_pools_.at(pool_id);
      std::lock_guard<std::mutex> g2(pool.mutex);
      TORCH_INTERNAL_ASSERT(pool.use_count > 0);
      pool.use_count++;
    }

    // TODO: I don't think this allows for doing multiple stream captures at once.
    // Actually, this is the exact issue I encountered when doing
    // conditional node stream capture. I should go ahead and just fix it here...
    captures_underway_.emplace_back(pool_id, std::move(filter));
  }

  void release_pool(std::pair<unsigned long long, unsigned long long> pool_id) {
    // I'm calling release_pool twice. Seems problematic!
    PrivatePool& pool = *cuda_mem_pools_.at(pool_id);
    int uc = 0;
    {
      std::lock_guard<std::mutex> g(m);
      // Problem: This design pattern prevents us from restarting
      // begin_allocate_to_pool multiple times, which is what
      // conditional nodes depend upon.
      uc = --pool.use_count;
    }

    if (uc == 0) {
      {
        std::lock_guard<std::mutex> g(pool.mutex);
        for (auto &&block: pool.blocks) {
          // TORCH_INTERNAL_ASSERT(block->allocated_); <- Incorrect. Suppose we free a block but then never try to allocate again. No time to move to move it to the free list.
          AT_CUDA_CHECK(cudaFreeHost(block->ptr));
          delete block;
        }

        for (auto &&free_list: pool.free_list_) {
          for (auto&& block: free_list.list_) {
            TORCH_INTERNAL_ASSERT(!block->allocated_);
            AT_CUDA_CHECK(cudaFreeHost(block->ptr));
            delete block;
          }
        }
      }

      std::lock_guard<std::mutex> g(m);
      cuda_mem_pools_.erase(pool_id);
    }
  }

  void end_allocate_to_pool(std::pair<unsigned long long, unsigned long long> pool_id) {
    free_finished_allocations(*cuda_mem_pools_.at(pool_id));

    std::lock_guard<std::mutex> g(m);
    auto it = std::find_if(captures_underway_.begin(), captures_underway_.end(),
        [pool_id](const auto& elem) {
            return std::get<0>(elem) == pool_id;
        });

    TORCH_INTERNAL_ASSERT(it != captures_underway_.end());
    captures_underway_.erase(it);
  }

  HostStats getStats() override {
    TORCH_INTERNAL_ASSERT(false, "getStats() not implemented yet");
  }

  void resetAccumulatedStats() override {
    TORCH_INTERNAL_ASSERT(false, "resetAccumulatedStats() not implemented yet");
  }

  void resetPeakStats() override {
    TORCH_INTERNAL_ASSERT(false, "resetAccumulatedStats() not implemented yet");
  }

  // Each capture id is unique, but theoretically there is no reason
  // why multiple stream captures to the same graph could not be
  // taking place at the same time.
  alignas(64) std::mutex m;
  std::vector<std::pair<MempoolId_t, std::function<bool(CUDAStream)>>> captures_underway_;
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash> cuda_mem_pools_;
};

CUDACachingHostAllocator caching_host_allocator;

CUDAStreamCapturableCachingHostAllocator stream_capturable_caching_host_allocator;

// Imagine if I could have a separate interface here, and return a
// different allocator when using stream capture. That would work very
// well.
static inline CachingHostAllocatorInterface<CUDAStream>& getCUDACachingHostAllocator() {
  cudaStreamCaptureStatus capture_status{cudaStreamCaptureStatusNone};
  AT_CUDA_CHECK(cudaStreamIsCapturing(getCurrentCUDAStream(), &capture_status));
  if (capture_status == cudaStreamCaptureStatusNone) {
    // std::cout << "GALVEZ:caching_host_allocator" << std::endl;
    return caching_host_allocator;
  } else {
    // std::cout << "GALVEZ:stream_capturable_caching_host_allocator" << std::endl;
    return stream_capturable_caching_host_allocator;
  }
}

void raw_local_deleter(void* ptr) {
  getCUDACachingHostAllocator().free(ptr);
}

void stream_capturable_raw_local_deleter(void* ptr) {
  stream_capturable_caching_host_allocator.free(ptr);
}

} // anonymous namespace

bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    at::cuda::CUDAStream stream) {
  return getCUDACachingHostAllocator().record_event(ptr, ctx, stream);
}

// Releases cached pinned memory allocations via cudaHostFree
void CachingHostAllocator_emptyCache() {
  getCUDACachingHostAllocator().empty_cache();
}

at::Allocator* getCachingHostAllocator() {
  return &getCUDACachingHostAllocator();
}

void CachingHostAllocator_beginAllocateToPool(
  MempoolId_t pool_id, std::function<bool(CUDAStream)> filter) {
  stream_capturable_caching_host_allocator.begin_allocate_to_pool(std::move(pool_id), std::move(filter));
}

void CachingHostAllocator_endAllocateToPool(
  MempoolId_t pool_id) {
  stream_capturable_caching_host_allocator.end_allocate_to_pool(std::move(pool_id));
}

void CachingHostAllocator_releasePool(
  MempoolId_t pool_id) {
  stream_capturable_caching_host_allocator.release_pool(std::move(pool_id));
}

at::HostStats CachingHostAllocator_getStats() {
  return getCUDACachingHostAllocator().getStats();
}

void CachingHostAllocator_resetAccumulatedStats() {
  return getCUDACachingHostAllocator().resetAccumulatedStats();
}

void CachingHostAllocator_resetPeakStats() {
  return getCUDACachingHostAllocator().resetPeakStats();
}

} // namespace at::cuda

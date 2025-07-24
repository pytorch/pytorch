#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <mutex>
#include <new>

namespace c10::CachingDeviceAllocator {

using namespace c10::CachingAllocator;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from device memory allocation.
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via device memory deallocation)
  StatArray inactive_split;

  // SUM: bytes allocated by this memory allocator
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

  // COUNT: total number of failed calls to device malloc necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to device memory allocation
  // after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // COUNT: total number of synchronize_and_free_events() calls
  int64_t num_sync_all_streams = 0;

  // COUNT: total number of device memory allocation calls. This includes both
  // mapped and malloced memory.
  int64_t num_device_alloc = 0;

  // COUNT: total number of device memory deallocation calls. This includes both
  // un-mapped and free memory.
  int64_t num_device_free = 0;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

} // namespace c10::CachingDeviceAllocator

namespace c10 {

using CaptureId_t = unsigned long long;

// first is set if the instance is created by Graph mode capture_begin.
// second is set if the instance is created by Graph mode graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

struct C10_API DeviceAllocator : public c10::Allocator {
  DeviceAllocator();
  ~DeviceAllocator() override;

  // Returns true if the allocator has been properly initialized and is ready
  // for use
  virtual bool initialized() = 0;

  // Releases all cached device memory from the specified memory pool back to
  // the system
  virtual void emptyCache(MempoolId_t mempool_id = {0, 0}) = 0;

  // Associates a memory allocation with a stream to establish dependency
  // tracking. Prevents memory reuse until all operations on the specified
  // stream complete
  virtual void recordStream(const DataPtr& ptr, c10::Stream stream) = 0;

  // Retrieves comprehensive memory statistics for the specified device,
  // including allocation patterns, usage metrics
  virtual CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) = 0;

  // Resets cumulative allocation statistics for the specified device to zero
  virtual void resetAccumulatedStats(c10::DeviceIndex device) = 0;

  // Resets peak memory usage statistics for the specified device
  virtual void resetPeakStats(c10::DeviceIndex device) = 0;

  // Return the free memory size and total memory size in bytes for the
  // specified device.
  virtual std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) = 0;
};

// This function is used to get the DeviceAllocator for a specific device type
// and keep backward compatibility with c10::GetAllocator.
C10_API inline DeviceAllocator* getDeviceAllocator(const DeviceType& t) {
  TORCH_CHECK(
      t != DeviceType::CPU,
      "getDeviceAllocator is not supported for CPU device type.");
  auto* allocator = c10::GetAllocator(t);
  auto* device_allocator = dynamic_cast<DeviceAllocator*>(allocator);
  TORCH_INTERNAL_ASSERT(
      device_allocator, "Allocator for ", t, " is not a DeviceAllocator.");
  return device_allocator;
}

constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks
const size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_destructive_interference_size;
#else
static constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

template <
    typename T,
    typename B,
    c10::DeleterFnPtr deleteFunc,
    typename BaseDeviceAllocator = c10::DeviceAllocator>
struct CachingDeviceAllocatorInterface : public BaseDeviceAllocator {
  B* get_allocated_block(void* ptr, bool remove = false) {
    const auto mutex_shard_id = get_mutex_shard_id(ptr);
    std::lock_guard<std::mutex> lock(mutex[mutex_shard_id].m);
    auto it = allocated_blocks[mutex_shard_id].find(ptr);
    if (it == allocated_blocks[mutex_shard_id].end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks[mutex_shard_id].erase(it);
    }
    return block;
  }

  at::DataPtr allocate(size_t size) override {
    c10::impl::VirtualGuardImpl impl(T::static_device_type);
    c10::Device device = impl.getDevice();
    void* devPtr = nullptr;
    c10::Stream stream = impl.getStream(device);
    T::allocate(&devPtr, size, device.index());

    if (size && TORCH_SDT_IS_ENABLED(malloc)) {
      TORCH_SDT_WITH_SEMAPHORE(malloc, devPtr, device, size, stream.id());
    }

    return {devPtr, devPtr, deleteFunc, device};
  }

  void malloc(void** devPtr, c10::DeviceIndex device, size_t size, S stream) {
    checkDeviceIndex(device);
    B* block = impls_[device]->malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = (void*)block->ptr;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_allocation(
          I::static_device_type, reinterpret_cast<uintptr_t>(*devPtr));
    }
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    B* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_deallocation(
          T::static_device_type, reinterpret_cast<uintptr_t>(block->ptr));
    }
    impls_[block->device]->free(block);
  }

  virtual DeleterFnPtr raw_deleter() const {
    return &deleteFunc;
  }

  void recordStream(const DataPtr& ptr, c10::Stream stream) override {
    if (!ptr.get()) {
      return;
    }
    if (ptr.get_deleter() != &deleteFunc) {
      return;
    }

    B* block = get_allocated_block(ptr.get());
    TORCH_CHECK(block, "No allocated block can be found.");
    device_allocators[block->device]->recordStream(block, stream);
  }

  void empty_cache(MempoolId_t mempool_id = {0, 0}) override {
    for (auto& impl : impls_) {
      impl->emptyCache(mempool_id);
    }
  }

  DeviceStats getDeviceStats(c10::DeviceIndex device) override {
    return impls_[device]->getDeviceStats();
  }

  void resetAccumulatedStats(c10::DeviceIndex device) override {
    checkDeviceIndex(device);
    impls_[device]->resetAccumulatedStats();
  }

  void resetPeakStats(c10::DeviceIndex device) override {
    checkDeviceIndex(device);
    impls_[device]->resetPeakStats();
  }

  // Initializes the per-device allocator implementations based on the given
  // device count. This method must be called before any allocator is used.
  void init(c10::DeviceIndex device_count) {
    const auto size = static_cast<c10::DeviceIndex>(impls_.size());
    if (size < device_count) {
      impls_.resize(device_count);
      for (const auto& i : c10::irange(size, device_count)) {
        impls_[i] = std::make_unique<T>(i);
      }
    }
  }

  bool initialized() override {
    return !impls_.empty();
  }

 private:
  void checkDeviceIndex(DeviceIndex device_index) const {
    TORCH_CHECK(
        0 <= device_index && device_index < impls_.size(),
        "Invalid device argument ",
        static_cast<int>(device_index),
        ": did you call init?");
  }

  void add_allocated_block(B* block) {
    const auto mutex_shard_id = get_mutex_shard_id(block->ptr);
    std::lock_guard<std::mutex> lock(mutex[mutex_shard_id].m);
    allocated_blocks[mutex_shard_id][block->ptr] = block;
  }

  static size_t get_mutex_shard_id(void* ptr) {
    return twang_mix64(static_cast<size_t>(ptr)) % kNumMutexShard;
  }

  // Aligns the mutex to avoid false sharing on cache lines.
  struct alignas(hardware_destructive_interference_size) AlignedMutex {
    std::mutex m;
  };
  // A prime number close to 64, used for sharding to reduce contention.
  static constexpr size_t kNumMutexShard = 67;
  std::array<AlignedMutex, kNumMutexShard> mutex;
  std::vector<std::unique_ptr<T>> impls_;
};

/**
 * Note [DeviceAllocator design]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 */

template <typename B>
struct BlockComparatorSize {
  bool operator()(const B* a, const B* b) const {
    if (a->size != b->size) {
      return a->size < b->size;
    }
    if (a->stream != b->stream) {
      return (uintptr_t)a->stream < (uintptr_t)b->stream;
    }
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
  }
};

template <typename B>
struct BlockComparatorAddress {
  bool operator()(const B* a, const B* b) const {
    if (a->stream != b->stream) {
      return (uintptr_t)a->stream < (uintptr_t)b->stream;
    }
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
  }
};

// Forward declaration
template <typename B>
class PrivatePool;

template <typename B>
struct BlockPool {
  BlockPool(bool small, PrivatePool<B>* private_pool = nullptr)
      : blocks(BlockComparatorSize<B>),
        unmapped(BlockComparatorAddress<B>),
        is_small(small),
        owner_PrivatePool(private_pool) {}

  // Do not insert a Block to blocks directly; use insert_into_blocks(),
  // instead.
  std::set<B*, BlockComparatorSize<B>> blocks;
  std::set<B*, BlockComparatorAddress<B>> unmapped;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool is_small;
  PrivatePool<B>* owner_PrivatePool;
  int64_t get_free_blocks_call_count{0};

  // Add a Block into blocks set with updating gc counter.
  std::pair<typename std::set<B*, BlockComparatorSize<B>>::iterator, bool>
  insert_into_blocks(B* block) {
    block->gc_count_base = get_free_blocks_call_count;
    return blocks.insert(block);
  }

  MempoolId_t owner_MempoolId() const {
    if (owner_PrivatePool) {
      return owner_PrivatePool->id;
    } else {
      return {0, 0};
    }
  }
};

struct ExpandableSegment;

template <typename S>
struct DeviceBlock {
  using Block = DeviceBlock<S>; // Convenience alias for self-reference

  DeviceBlock(
      c10::DeviceIndex device,
      S stream,
      size_t size,
      BlockPool<Block>* pool,
      void* ptr)
      : device(device),
        stream(stream),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // constructor for search key
  DeviceBlock(c10::DeviceIndex device, S stream, size_t size)
      : device(device), stream(stream), size(size), requested_size(0) {}

  size_t gc_count() const {
    TORCH_INTERNAL_ASSERT(pool);
    return static_cast<int>(pool->get_free_blocks_call_count - gc_count_base);
  }

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }

  void splice(Block* before, Block* after) {
    if (before) {
      TORCH_INTERNAL_ASSERT(before->next == after);
      before->next = this;
    }
    prev = before;
    if (after) {
      TORCH_INTERNAL_ASSERT(after->prev == before);
      after->prev = this;
    }
    next = after;
  }

  c10::DeviceIndex device; // gpu
  S stream; // allocation stream
  ska::flat_hash_set<S> stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  size_t requested_size; // memory originally requested
  BlockPool<DeviceBlock<S>>* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in-use flag
  bool mapped{true}; // is the virtual address range this Block references
  // backed by physical pages. Always true when
  // expandable_segment_ is null. When false
  // This Block will be aligned to the segment size
  // of its expandable_segment_.
  Block* prev{nullptr}; // prev block if split from a larger allocation
  Block* next{nullptr}; // next block if split from a larger allocation
  int event_count{0}; // number of outstanding CUDA events
  int64_t gc_count_base{0}; // get_free_blocks_call_count when Block is inserted
  std::shared_ptr<GatheredContext> context_when_allocated;
  // only set for the first block in the segment (when prev == null)
  // this records the frame information when cudaMalloc was called
  // whereas context_when_allocated records the last time we handed this
  // memory out from our cache.
  std::shared_ptr<GatheredContext> context_when_segment_allocated;

  ExpandableSegment* expandable_segment_{nullptr};
};

template <typename B>
struct PrivatePool {
  PrivatePool(MempoolId_t id, DeviceAllocator* allocator = nullptr)
      : id(std::move(id)),
        allocator_(allocator),
        large_blocks(/*small=*/false, this),
        small_blocks(/*small=*/true, this) {}
  C10_DISABLE_COPY_AND_ASSIGN(PrivatePool);
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(PrivatePool&&) = delete;
  ~PrivatePool() = default;

  MempoolId_t id{0, 0};
  // Number of live graphs using this pool
  int use_count{1};
  // Number of unfreed device allocation made for this pool. When use_count and
  // deviceMalloc_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int deviceMalloc_count{0};
  // Instead of maintaining private BlockPools here, I could stuff all blocks
  // (private or no) into the top-level large_blocks and small_blocks, and
  // distinguish private blocks by adding a "pool id" check above the stream
  // check in BlockComparator. BlockComparator is performance- critical though,
  // I'd rather not add more logic to it.
  DeviceAllocator* allocator_;
  BlockPool<B> large_blocks;
  BlockPool<B> small_blocks;

 public:
  DeviceAllocator* allocator() {
    return allocator_;
  }
};

// Represents a contiguous virtual memory segment mapped for allocation.
struct SegmentRange {
  SegmentRange(void* p, size_t s, bool is_small)
      : ptr_(static_cast<char*>(p)),
        size_(s),
        // Use 2 MiB pages for small segments, 20 MiB pages for large segments.
        segment_size_(is_small ? kSmallBuffer : kLargeBuffer) {}

  template <bool is_open_interval = true>
  size_t begin() const {
    if constexpr (is_open_interval) {
      return segmentLeft(ptr_);
    } else {
      return segmentRight(ptr_);
    }
  }

  template <bool is_open_interval = true>
  size_t end() const {
    if constexpr (is_open_interval) {
      return segmentRight(ptr_ + size_);
    } else {
      return segmentLeft(ptr_ + size_);
    }
  }

private:
  char* ptr_;          // Starting address of the mapped range.
  size_t size_;        // Size in bytes of the mapped range.
  size_t segment_size_; // Page size used for the segment (2 MiB or 20 MiB).
};

struct ExpandableSegment {
  virtual ExpandableSegment(
      c10::DeviceIndex device,
      std::optional<c10::Stream> stream,
      size_t address_space_size,
      size_t segment_size,
      std::vector<c10::DeviceIndex> peers) = 0;
  C10_DISABLE_COPY_AND_ASSIGN(ExpandableSegment);
  ExpandableSegment(ExpandableSegment&&) = delete;
  ExpandableSegment operator=(ExpandableSegment&&) = delete;

  // Maps a virtual memory range to physical memory.
  //
  // `range.start` must be aligned to `segment_size_`.
  // The returned range may be larger than requested if `range.size` is not
  // aligned to `segment_size_`.
  //
  // Return:
  // - A valid `SegmentRange` representing the actual mapped memory.
  // - If the return range has `.size() == 0`, it indicates out-of-memory (OOM).
  //
  // Must be implemented by subclasses.
  virtual SegmentRange map(SegmentRange range) = 0;

  virtual SegmentRange unmap(SegmentRange range) {
    auto begin = segmentRight(range.ptr);
    auto end = segmentLeft(range.ptr + range.size);
    if (begin >= end) {
      return SegmentRange{range.ptr, 0};
    }
    unmapHandles(begin, end);
    return rangeFromHandles(begin, end);
  }

 private:
  char* ptr() const {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<char*>(ptr_);
  }

  size_t size() const {
    return max_handles_ * segment_size_;
  }

  void addPeer(c10::DeviceIndex device) {
    peers_.push_back(device);
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { setAccess(device, begin, end); });
  }

  ~ExpandableSegment() {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressFree_(
        ptr_, segment_size_ * max_handles_));
  }

 private:
  void setAccess(c10::DeviceIndex device, size_t begin, size_t end) {
    CUmemAccessDesc desc;
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    desc.location.id = static_cast<int>(device);
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemSetAccess_(
        ptr_ + begin * segment_size_, (end - begin) * segment_size_, &desc, 1));
  }

  void mapAndSetAccess(size_t begin, size_t end) {
    for (auto i : c10::irange(begin, end)) {
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemMap_(
          ptr_ + i * segment_size_,
          segment_size_,
          0,
          // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
          handles_.at(i).value().handle,
          0ULL));
    }
    setAccess(device_, begin, end);
    for (auto p : peers_) {
      setAccess(p, begin, end);
    }
  }

  void unmapHandles(size_t begin, size_t end) {
    // note: unlike cudaFree, MemUnmap and MemRelease do
    // not appear to synchronize in all cases, so we have to wait for the
    // stream to finish before this memory is truly free.

    // cannot call c10::cuda::stream_synchronize because
    // it might grab the GIL which can lead to a deadlock
    // Locking order must be GIL -> Allocator Lock
    if (stream_) {
      C10_CUDA_CHECK(cudaStreamSynchronize(*stream_));
    } else {
      cuda::CUDAGuard device_guard(device_);
      C10_CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (auto i : c10::irange(begin, end)) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      Handle h = handles_.at(i).value();
      handles_.at(i) = std::nullopt;
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemUnmap_(
          ptr_ + segment_size_ * i, segment_size_));
      if (h.shareable_handle) {
        close(std::get<int>(*h.shareable_handle));
      }
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h.handle));
    }
    trimHandles();
  }
  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }
  void forEachAllocatedRange(const std::function<void(size_t, size_t)>& fn) {
    size_t start = 0;
    for (auto i : c10::irange(handles_.size())) {
      if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
        start = i;
      }
      if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
        fn(start, i + 1);
      }
    }
  }
  size_t numSegments(size_t size) {
    return (size + segment_size_ - 1) / segment_size_;
  }
  size_t segmentLeft(char* p) {
    auto size = p - ptr();
    return size / segment_size_;
  }
  size_t segmentRight(char* p) {
    auto size = p - ptr();
    return numSegments(size);
  }
  SegmentRange rangeFromHandles(size_t begin, size_t end) {
    return SegmentRange(
        ptr() + segment_size_ * begin, segment_size_ * (end - begin));
  }
  c10::DeviceIndex device_;
  std::optional<cudaStream_t> stream_;
  CUdeviceptr ptr_{};
  size_t segment_size_;
  size_t max_handles_;
  struct Handle {
    CUmemGenericAllocationHandle handle;
    std::optional<std::variant<int, CUmemFabricHandle>> shareable_handle;
  };
  std::vector<std::optional<Handle>> handles_;
  // devices on which this memory should be mapped in addition
  // to the device where the physical memory lives (device_).
  std::vector<c10::DeviceIndex> peers_;
};

template <typename S, typename E, typename B = DeviceBlock<S>>
struct CachingDeviceAllocatorImpl {
  virtual ~CachingDeviceAllocatorImpl() = default;

 public:
 private:
 protected:
};

} // namespace c10

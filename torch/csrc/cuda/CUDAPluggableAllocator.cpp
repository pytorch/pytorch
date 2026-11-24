#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

namespace torch::cuda::CUDAPluggableAllocator {

int device_count = 0;

void custom_raw_deleter(void* ptr);

_AllocationMetadata::_AllocationMetadata() : size(0), device_idx(-1) {}

_AllocationMetadata::_AllocationMetadata(
    size_t size,
    c10::DeviceIndex device_idx,
    cudaStream_t stream)
    : size(size), device_idx(device_idx), stream(stream) {}

// This is a fast API to just register allocators
// based on function pointers (ie. external .so libraries)
// This avoids having to link against libtorch for C++ based custom allocators
// And also use this from python
CUDAPluggableAllocator::CUDAPluggableAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn)
    : alloc_fn_(std::move(alloc_fn)), free_fn_(std::move(free_fn)) {}

CUDAPluggableAllocator::CUDAPluggableAllocator(CUDAPluggableAllocator& other)
    : alloc_fn_(other.alloc_fn_),
      free_fn_(other.free_fn_),
      init_fn_(other.init_fn_),
      reset_fn_(other.reset_fn_),
      memory_fraction_fn_(other.memory_fraction_fn_),
      base_alloc_fn_(other.base_alloc_fn_),
      record_stream_fn_(other.record_stream_fn_),
      begin_allocate_to_pool_fn_(other.begin_allocate_to_pool_fn_),
      end_allocate_to_pool_fn_(other.end_allocate_to_pool_fn_),
      relase_pool_fn_(other.relase_pool_fn_) {}

void CUDAPluggableAllocator::set_init_fn(std::function<void(int)> init_fn) {
  init_fn_ = std::move(init_fn);
}

void CUDAPluggableAllocator::set_reset_fn(std::function<void()> reset_fn) {
  reset_fn_ = std::move(reset_fn);
}

void CUDAPluggableAllocator::set_memory_fraction_fn(
    std::function<void(double, int)> memory_fraction_fn) {
  memory_fraction_fn_ = std::move(memory_fraction_fn);
}

void CUDAPluggableAllocator::set_base_alloc_fn(
    std::function<void*(void*, size_t*)> base_alloc_fn) {
  base_alloc_fn_ = std::move(base_alloc_fn);
}

void CUDAPluggableAllocator::set_record_stream_fn(
    std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn) {
  record_stream_fn_ = std::move(record_stream_fn);
}

void CUDAPluggableAllocator::set_begin_allocate_to_pool(
    std::function<
        void(int, c10::cuda::MempoolId_t, std::function<bool(cudaStream_t)>)>
        capture_begin_fn) {
  begin_allocate_to_pool_fn_ = std::move(capture_begin_fn);
}

void CUDAPluggableAllocator::set_end_allocate_to_pool_fn(
    std::function<void(int, c10::cuda::MempoolId_t)> capture_about_to_end_fn) {
  end_allocate_to_pool_fn_ = std::move(capture_about_to_end_fn);
}

void CUDAPluggableAllocator::set_release_pool(
    std::function<void(int, c10::cuda::MempoolId_t)> capture_destroy_fn) {
  relase_pool_fn_ = std::move(capture_destroy_fn);
}

void* CUDAPluggableAllocator::malloc(
    size_t size,
    c10::DeviceIndex device,
    cudaStream_t stream) {
  void* r = alloc_fn_(size, device, stream);
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream));
  }
  return r;
}

c10::DataPtr CUDAPluggableAllocator::allocate(size_t size) {
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  void* r = this->malloc(size, device, stream);
  c10::DataPtr data_ptr = {
      r, r, raw_deleter(), c10::Device(c10::DeviceType::CUDA, device)};
  return data_ptr;
}

c10::DeleterFnPtr CUDAPluggableAllocator::raw_deleter() const {
  return &custom_raw_deleter;
}

void* CUDAPluggableAllocator::raw_alloc(size_t nbytes) {
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  return malloc(nbytes, device, stream);
}

void* CUDAPluggableAllocator::raw_alloc_with_stream(
    size_t nbytes,
    cudaStream_t stream) {
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  return malloc(nbytes, device, stream);
}

void CUDAPluggableAllocator::raw_delete(void* ptr) {
  cudaStream_t stream{};
  c10::DeviceIndex device_idx = -1;
  size_t size = 0;
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    TORCH_CHECK(
        allocation_metadata_.count(ptr),
        "Trying to free a pointer not allocated here");
    _AllocationMetadata& metadata = allocation_metadata_[ptr];
    size = metadata.size;
    device_idx = metadata.device_idx;
    stream = metadata.stream;
    allocation_metadata_.erase(ptr);
  }
  free_fn_(ptr, size, device_idx, stream);
}

void CUDAPluggableAllocator::init(int device_count) {
  if (init_fn_) {
    init_fn_(device_count);
  }
  initialized_ = true;
}

bool CUDAPluggableAllocator::initialized() {
  return initialized_;
}

double CUDAPluggableAllocator::getMemoryFraction(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getMemoryFraction. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::setMemoryFraction(
    double fraction,
    c10::DeviceIndex device) {
  if (memory_fraction_fn_) {
    memory_fraction_fn_(fraction, device);
  }
}

std::vector<c10::cuda::CUDACachingAllocator::StreamSegmentSize>
CUDAPluggableAllocator::getExpandableSegmentSizes(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "CUDAMallocAsyncAllocator does not yet support getExpandableSegmentSizes.");
}

void CUDAPluggableAllocator::emptyCache(
    /*unused*/ c10::cuda::MempoolId_t mempool_id) {
  if (reset_fn_) {
    reset_fn_();
  }
}

void CUDAPluggableAllocator::cacheInfo(
    c10::DeviceIndex device,
    size_t* largestBlock) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support cacheInfo. "
      "If you need it, please file an issue describing your use case.");
}

void* CUDAPluggableAllocator::getBaseAllocation(void* ptr, size_t* size) {
  if (base_alloc_fn_) {
    return base_alloc_fn_(ptr, size);
  } else {
    return ptr;
  }
}

void CUDAPluggableAllocator::recordStream(
    const c10::DataPtr& ptr,
    c10::cuda::CUDAStream stream) {
  if (record_stream_fn_) {
    record_stream_fn_(ptr.get(), stream);
  }
}

c10::CachingDeviceAllocator::DeviceStats CUDAPluggableAllocator::getDeviceStats(
    c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getDeviceStats. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support resetAccumulatedStats. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::resetPeakStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support resetPeakStats. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::SnapshotInfo CUDAPluggableAllocator::snapshot(
    c10::cuda::MempoolId_t mempool_id,
    bool include_traces) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support snapshot. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::ShareableHandle CUDAPluggableAllocator::
    shareIpcHandle(void* ptr) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support shareIPcHandle. "
      "If you need it, please file an issue describing your use case.");
}

std::shared_ptr<void> CUDAPluggableAllocator::getIpcDevPtr(std::string handle) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getIpcDevPtr. "
      "If you need it, please file an issue describing your use case.");
}

// CUDAGraph interactions
void CUDAPluggableAllocator::beginAllocateToPool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id,
    std::function<bool(cudaStream_t)> filter) {
  if (begin_allocate_to_pool_fn_) {
    begin_allocate_to_pool_fn_(device, mempool_id, std::move(filter));
  }
}

void CUDAPluggableAllocator::endAllocateToPool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id) {
  if (end_allocate_to_pool_fn_) {
    end_allocate_to_pool_fn_(device, mempool_id);
  }
}

void CUDAPluggableAllocator::releasePool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id) {
  if (relase_pool_fn_) {
    relase_pool_fn_(device, mempool_id);
  }
}

void CUDAPluggableAllocator::recordHistory(
    bool enabled,
    c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    c10::cuda::CUDACachingAllocator::RecordContext when,
    bool clearHistory,
    const std::vector<std::string>& skip_actions) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support recordHistory. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::attachOutOfMemoryObserver(
    c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support attachOutOfMemoryObserver. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::attachOomRejectionObserver(
    c10::cuda::CUDACachingAllocator::OomRejectionObserver observer) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "CUDAPluggableAllocator does not yet support attachOomRejectionObserver. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::attachAllocatorTraceTracker(
    c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not support attachAllocatorTraceTracker. "
      "attachAllocatorTraceTracker is only used inside Pytorch.");
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
CUDAPluggableAllocator::getCheckpointState(
    c10::DeviceIndex device,
    at::cuda::MempoolId_t id) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getCheckpointState. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::CheckpointDelta CUDAPluggableAllocator::
    setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support setCheckpointPoolState. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  c10::cuda::CUDAGuard device_guard(dev);
  cudaError_t err = cudaDeviceEnablePeerAccess(dev_to_access, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    // ignore and clear the error if access was already enabled
    (void)cudaGetLastError();
  } else {
    C10_CUDA_CHECK(err);
  }
}

cudaError_t CUDAPluggableAllocator::memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    cudaStream_t stream,
    bool p2p_enabled) {
  return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

std::string CUDAPluggableAllocator::name() {
  return "pluggable";
}

void CUDAPluggableAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  C10_CUDA_CHECK(cudaMemcpyAsync(
      dest,
      src,
      count,
      cudaMemcpyDeviceToDevice,
      c10::cuda::getCurrentCUDAStream()));
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
    current_custom_allocator;

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
getCurrentAllocator() {
  return current_custom_allocator;
}

// TODO: add more functions in the argument
std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
createCustomAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn) {
  std::shared_ptr<CUDAPluggableAllocator> allocator(
      new CUDAPluggableAllocator(std::move(alloc_fn), std::move(free_fn)));
  allocator->init(device_count);
  return allocator;
}

void changeCurrentAllocator(
    const std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>&
        allocator) {
  TORCH_CHECK(
      !c10::cuda::CUDACachingAllocator::allocator.load()->initialized(),
      "Can't swap an already initialized allocator");
  c10::cuda::CUDACachingAllocator::allocator.store(allocator.get());
  current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ptr) {
  current_custom_allocator->raw_delete(ptr);
}

namespace {

namespace CCA = c10::cuda::CUDACachingAllocator;
using c10::cuda::CaptureId_t;
using c10::cuda::MempoolId_t;

// The DataPtr context of a tensor made by a window's allocator.
struct Owned {
  std::shared_ptr<CCA::CUDAAllocator> allocator;
  void* ptr;
};

void deleteOwned(void* ctx);

// A layer over the native caching allocator (inner). A graph allocator window
// (beginGraphAllocatorWindow, opened inside a CUDA graph capture) makes it the
// current CUDA allocator until the window ends. It forwards everything to
// inner, except the tensor allocations that inner would place into the pool of
// the window's capture: the window's allocator makes those, one call per
// allocation at its exact size.
class LayeredAllocator final : public CCA::CUDAAllocator {
 public:
  void beginWindow(std::shared_ptr<CCA::CUDAAllocator> allocator) {
    TORCH_CHECK(allocator, "a CUDA graph allocator window needs an allocator");
    const c10::DeviceIndex device = c10::cuda::current_device();
    const auto capture = c10::cuda::captureIdMayInitCtx(
        c10::cuda::getCurrentCUDAStream(device).stream());
    TORCH_CHECK(
        capture.has_value(),
        "a CUDA graph allocator window must be opened inside a CUDA graph "
        "capture, on its capturing stream");
    std::lock_guard<std::mutex> lock(mutex_);
    auto* current = CCA::allocator.load();
    TORCH_CHECK(
        current != this, "a CUDA graph allocator window is already open");
    // The layer forwards to the caching allocator, which keeps owning
    // everything allocated so far.
    TORCH_CHECK(
        current->name() == "native",
        "CUDA graph allocator windows require the native CUDA caching "
        "allocator, but the current allocator is '",
        current->name(),
        "'");
    inner_.store(current);
    window_ = Window{std::move(allocator), *capture, device};
    active_.store(true, std::memory_order_relaxed);
    CCA::allocator.store(this);
  }
  // Idempotent. Returns whether another capture began or ended during the
  // window.
  bool endWindow() {
    std::optional<Window> released; // dropped after the lock: may run Python
    std::lock_guard<std::mutex> lock(mutex_);
    if (CCA::allocator.load() == this) {
      CCA::allocator.store(inner());
    }
    active_.store(false, std::memory_order_relaxed);
    released = std::move(window_);
    window_.reset();
    // a scope still open is older than the next window's capture
    scopes_.clear();
    return other_capture_.exchange(false);
  }

  c10::DataPtr allocate(size_t size) override {
    if (auto c = claimant(size)) {
      void* ptr = allocFrom(*c, size);
      return {
          ptr,
          new Owned{std::move(c->allocator), ptr},
          &deleteOwned,
          c10::Device(c10::DeviceType::CUDA, c->device)};
    }
    return inner()->allocate(size);
  }
  void recordStream(const c10::DataPtr& ptr, c10::cuda::CUDAStream stream)
      override {
    // A window allocator's tensor: its allocator handles the stream use (a
    // CUDAPluggableAllocator passes it to its record_stream_fn, if set). After
    // the window, the caching allocator ignores memory it does not own.
    if (ptr.get_deleter() == &deleteOwned) {
      static_cast<Owned*>(ptr.get_context())
          ->allocator->recordStream(ptr, stream);
      return;
    }
    inner()->recordStream(ptr, stream);
  }

  // Pool scopes opened during the window (use_mem_pool, NCCL and symmetric
  // memory pools) are newer than the capture's own scope, so they win over it,
  // as in inner's newest-first search: the layer keeps them to answer the same.
  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)> filter) override {
    inner()->beginAllocateToPool(device, mempool_id, filter);
    std::lock_guard<std::mutex> lock(mutex_);
    if (window_) {
      scopes_.push_back(Scope{device, mempool_id, std::move(filter)});
    }
  }
  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id)
      override {
    inner()->endAllocateToPool(device, mempool_id);
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      if (it->device == device && it->pool == mempool_id) {
        scopes_.erase(std::next(it).base());
        break;
      }
    }
  }
  // These reach the layer only when another capture (e.g. a conditional
  // node's body) begins or ends while it is current: CUDAGraphMemory sends the
  // registration-aware hooks only to the native allocator. Not supported
  // during a window: forwarded, so that the caching allocator's count of
  // captures stays balanced, and reported when the window ends (raising here
  // would leave the capture half begun).
  void markCaptureBegin(c10::DeviceIndex device) override {
    other_capture_.store(true);
    inner()->markCaptureBegin(device);
  }
  void markCaptureEnd(c10::DeviceIndex device) override {
    other_capture_.store(true);
    inner()->markCaptureEnd(device);
  }

  // Everything else is inner's. A virtual added to CUDAAllocator (or its
  // bases) must be forwarded here, or the layer silently gets its default.
  using MemoryInfo = std::pair<size_t, size_t>;
#define LAYERED_FORWARD(ret, name, params, args) \
  ret name params override {                     \
    return inner()->name args;                   \
  }
  LAYERED_FORWARD(void*, raw_alloc, (size_t nbytes), (nbytes))
  LAYERED_FORWARD(
      void*,
      raw_alloc_with_stream,
      (size_t nbytes, cudaStream_t stream),
      (nbytes, stream))
  LAYERED_FORWARD(void, raw_delete, (void* ptr), (ptr))
  LAYERED_FORWARD(c10::DeleterFnPtr, raw_deleter, () const, ())
  LAYERED_FORWARD(
      void,
      copy_data,
      (void* dest, const void* src, std::size_t count) const,
      (dest, src, count))
  LAYERED_FORWARD(
      std::optional<c10::AllocationStreamInfo>,
      allocation_stream_info,
      (const void* ptr) const,
      (ptr))
  LAYERED_FORWARD(bool, initialized, (), ())
  LAYERED_FORWARD(void, emptyCache, (MempoolId_t mempool_id), (mempool_id))
  LAYERED_FORWARD(
      c10::CachingDeviceAllocator::DeviceStats,
      getDeviceStats,
      (c10::DeviceIndex device),
      (device))
  LAYERED_FORWARD(
      void,
      resetAccumulatedStats,
      (c10::DeviceIndex device),
      (device))
  LAYERED_FORWARD(void, resetPeakStats, (c10::DeviceIndex device), (device))
  LAYERED_FORWARD(void, init, (int device_count), (device_count))
  LAYERED_FORWARD(
      double,
      getMemoryFraction,
      (c10::DeviceIndex device),
      (device))
  LAYERED_FORWARD(
      void,
      setMemoryFraction,
      (double fraction, c10::DeviceIndex device),
      (fraction, device))
  LAYERED_FORWARD(
      std::vector<CCA::StreamSegmentSize>,
      getExpandableSegmentSizes,
      (c10::DeviceIndex device),
      (device))
  LAYERED_FORWARD(void, enable, (bool value), (value))
  LAYERED_FORWARD(bool, isEnabled, () const, ())
  LAYERED_FORWARD(
      void,
      cacheInfo,
      (c10::DeviceIndex device, size_t* largestBlock),
      (device, largestBlock))
  LAYERED_FORWARD(
      void*,
      getBaseAllocation,
      (void* ptr, size_t* size),
      (ptr, size))
  LAYERED_FORWARD(
      CCA::SnapshotInfo,
      snapshot,
      (MempoolId_t mempool_id, bool include_traces),
      (mempool_id, include_traces))
  LAYERED_FORWARD(bool, isCaptureContext, (c10::DeviceIndex device), (device))
  LAYERED_FORWARD(
      void,
      releasePool,
      (c10::DeviceIndex device, MempoolId_t mempool_id),
      (device, mempool_id))
  LAYERED_FORWARD(
      int,
      getPoolUseCount,
      (c10::DeviceIndex device, MempoolId_t mempool_id),
      (device, mempool_id))
  LAYERED_FORWARD(
      void,
      createOrIncrefPool,
      (c10::DeviceIndex device,
       MempoolId_t mempool_id,
       std::shared_ptr<CCA::CUDAAllocator> allocator),
      (device, mempool_id, std::move(allocator)))
  LAYERED_FORWARD(
      void,
      setUseOnOOM,
      (c10::DeviceIndex device, MempoolId_t mempool_id, bool use_on_oom),
      (device, mempool_id, use_on_oom))
  LAYERED_FORWARD(
      void,
      setNoSplit,
      (c10::DeviceIndex device, MempoolId_t mempool_id),
      (device, mempool_id))
  LAYERED_FORWARD(
      bool,
      checkPoolLiveAllocations,
      (c10::DeviceIndex device,
       MempoolId_t mempool_id,
       const std::unordered_set<void*>& expected_live_allocations),
      (device, mempool_id, expected_live_allocations))
  LAYERED_FORWARD(CCA::ShareableHandle, shareIpcHandle, (void* ptr), (ptr))
  LAYERED_FORWARD(
      std::shared_ptr<void>,
      getIpcDevPtr,
      (std::string handle),
      (std::move(handle)))
  LAYERED_FORWARD(bool, isHistoryEnabled, (), ())
  LAYERED_FORWARD(
      std::shared_ptr<c10::GatheredContext>,
      getContextForPointer,
      (const void* ptr),
      (ptr))
  LAYERED_FORWARD(
      void,
      recordHistory,
      (bool enabled,
       CCA::CreateContextFn context_recorder,
       size_t alloc_trace_max_entries,
       CCA::RecordContext when,
       bool clearHistory,
       const std::vector<std::string>& skip_actions),
      (enabled,
       std::move(context_recorder),
       alloc_trace_max_entries,
       when,
       clearHistory,
       skip_actions))
  LAYERED_FORWARD(
      void,
      recordAnnotation,
      (const std::vector<std::pair<std::string, std::string>>& md),
      (md))
  LAYERED_FORWARD(void, pushCompileContext, (std::string & md), (md))
  LAYERED_FORWARD(void, popCompileContext, (), ())
  LAYERED_FORWARD(bool, supportsUserMetadata, (), ())
  LAYERED_FORWARD(
      void,
      setUserMetadata,
      (const std::string& metadata),
      (metadata))
  LAYERED_FORWARD(std::string, getUserMetadata, (), ())
  LAYERED_FORWARD(
      void,
      annotateMemory,
      (const void* ptr, const std::string& metadata),
      (ptr, metadata))
  LAYERED_FORWARD(
      void,
      attachOutOfMemoryObserver,
      (CCA::OutOfMemoryObserver observer),
      (std::move(observer)))
  LAYERED_FORWARD(
      void,
      attachOomRejectionObserver,
      (CCA::OomRejectionObserver observer),
      (std::move(observer)))
  LAYERED_FORWARD(
      void,
      attachAllocatorTraceTracker,
      (CCA::AllocatorTraceTracker tracker),
      (std::move(tracker)))
  LAYERED_FORWARD(
      void,
      enablePeerAccess,
      (c10::DeviceIndex dev, c10::DeviceIndex dev_to_access),
      (dev, dev_to_access))
  LAYERED_FORWARD(
      cudaError_t,
      memcpyAsync,
      (void* dst,
       int dstDevice,
       const void* src,
       int srcDevice,
       size_t count,
       cudaStream_t stream,
       bool p2p_enabled),
      (dst, dstDevice, src, srcDevice, count, stream, p2p_enabled))
  LAYERED_FORWARD(
      std::shared_ptr<CCA::AllocatorState>,
      getCheckpointState,
      (c10::DeviceIndex device, MempoolId_t id),
      (device, id))
  LAYERED_FORWARD(
      CCA::CheckpointDelta,
      setCheckpointPoolState,
      (c10::DeviceIndex device, std::shared_ptr<CCA::AllocatorState> pps),
      (device, std::move(pps)))
  LAYERED_FORWARD(
      void,
      restoreExpandableSegment,
      (c10::DeviceIndex device,
       cudaStream_t stream,
       MempoolId_t mempool_id,
       bool is_small,
       size_t address,
       size_t reserve_size,
       size_t segment_size,
       CCA::Expandable_Segments_Handle_Type handle_type,
       const std::vector<std::pair<size_t, size_t>>& mapped_ranges),
      (device,
       stream,
       mempool_id,
       is_small,
       address,
       reserve_size,
       segment_size,
       handle_type,
       mapped_ranges))
  LAYERED_FORWARD(
      c10::DataPtr,
      allocateWithAddress,
      (size_t size, void* addr),
      (size, addr))
  LAYERED_FORWARD(std::string, name, (), ())
  LAYERED_FORWARD(
      MemoryInfo,
      getMemoryInfo,
      (c10::DeviceIndex device),
      (device))
  LAYERED_FORWARD(
      bool,
      is_simple_data_ptr,
      (const c10::DataPtr& data_ptr) const,
      (data_ptr))
#undef LAYERED_FORWARD

 private:
  struct Window {
    std::shared_ptr<CCA::CUDAAllocator> allocator;
    CaptureId_t capture;
    c10::DeviceIndex device;
  };
  struct Scope {
    c10::DeviceIndex device;
    MempoolId_t pool;
    std::function<bool(cudaStream_t)> filter;
  };
  struct Claimed {
    std::shared_ptr<CCA::CUDAAllocator> allocator;
    c10::DeviceIndex device;
    cudaStream_t stream;
  };

  CCA::CUDAAllocator* inner() const {
    return inner_.load(std::memory_order_relaxed);
  }

  // The window's allocator, if inner would place an allocation of `size` on
  // the current stream of the current device into the pool of the window's
  // capture: no newer pool scope takes it, and the stream is capturing into
  // that capture (the capture's own scope filter).
  std::optional<Claimed> claimant(size_t size) {
    if (C10_LIKELY(!active_.load(std::memory_order_relaxed)) || size == 0) {
      return std::nullopt;
    }
    const c10::DeviceIndex device = c10::cuda::current_device();
    const cudaStream_t s = c10::cuda::getCurrentCUDAStream(device).stream();
    std::lock_guard<std::mutex> lock(mutex_);
    if (!window_ || window_->device != device) {
      return std::nullopt;
    }
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      if (it->device == device && it->filter(s)) {
        return std::nullopt;
      }
    }
    const auto capture = c10::cuda::captureIdMayInitCtx(s);
    if (!capture.has_value() || *capture != window_->capture) {
      return std::nullopt;
    }
    return Claimed{window_->allocator, device, s};
  }

  static void* allocFrom(const Claimed& c, size_t size) {
    void* ptr = nullptr;
    {
      // as for the caching allocator's own segments during a capture
      // (cudaMallocMaybeCapturing): the allocator may cudaMalloc
      c10::cuda::CUDAStreamCaptureModeGuard mode{cudaStreamCaptureModeRelaxed};
      ptr = c.allocator->raw_alloc_with_stream(size, c.stream);
    }
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        ptr != nullptr,
        "CUDA out of memory: the CUDA graph allocator returned null for ",
        size,
        " bytes");
    if (C10_UNLIKELY(reinterpret_cast<uintptr_t>(ptr) % 256 != 0)) {
      c.allocator->raw_delete(ptr);
      TORCH_CHECK(
          false,
          "The CUDA graph allocator returned ",
          ptr,
          ", which is not aligned to 256 bytes");
    }
    return ptr;
  }

  std::atomic<CCA::CUDAAllocator*> inner_{nullptr};
  std::atomic<bool> active_{false};
  std::atomic<bool> other_capture_{false};
  std::mutex mutex_;
  std::optional<Window> window_;
  std::vector<Scope> scopes_;
};

// Current only while a graph allocator window is open (a planned capture
// runs). Never deleted, because storages created during a window keep a raw
// pointer to it (the resize rule dynamic_casts it, and storage ops like
// _lazy_clone call it).
LayeredAllocator& layered() {
  static auto* allocator = new LayeredAllocator();
  return *allocator;
}

void deleteOwned(void* ctx) {
  std::unique_ptr<Owned> owned(static_cast<Owned*>(ctx));
  owned->allocator->raw_delete(owned->ptr);
}

} // namespace

void beginGraphAllocatorWindow(
    std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator) {
  layered().beginWindow(std::move(allocator));
}

bool endGraphAllocatorWindow() {
  return layered().endWindow();
}

} // namespace torch::cuda::CUDAPluggableAllocator

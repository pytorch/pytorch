#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_set>
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

bool graphAllocatorsEnabled();

void changeCurrentAllocator(
    const std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>&
        allocator) {
  TORCH_CHECK(
      !graphAllocatorsEnabled(),
      "change_current_allocator cannot be used once CUDA graph allocators "
      "are enabled");
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
using PoolKey = std::pair<c10::DeviceIndex, MempoolId_t>;

// The DataPtr context of a tensor made by a capture allocator.
struct Owned {
  std::shared_ptr<CCA::CUDAAllocator> allocator;
  void* ptr;
};

void deleteOwned(void* ctx);

// The current CUDA allocator after torch.cuda.memory._enable_graph_allocators.
// It forwards everything to the native caching allocator (inner), except the
// allocations inner would place into the pool of a capture given an allocator
// (CUDAGraph::capture_begin(..., allocator)) while that capture is in
// progress: the capture's allocator makes those, one call per allocation at
// its exact size.
class LayeredAllocator final : public CCA::CUDAAllocator {
 public:
  void install(CCA::CUDAAllocator* inner) {
    inner_.store(inner);
    CCA::allocator.store(this);
  }
  bool installed() const {
    return CCA::allocator.load() == this;
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
    // A capture allocator's tensor: its allocator handles the stream use (a
    // CUDAPluggableAllocator passes it to its record_stream_fn, if set), as
    // for any allocator's own tensors.
    if (ptr.get_deleter() == &deleteOwned) {
      static_cast<Owned*>(ptr.get_context())
          ->allocator->recordStream(ptr, stream);
      return;
    }
    inner()->recordStream(ptr, stream);
  }

  // Called by CUDAGraph::capture_begin once the capture has begun, with the
  // pool's allocation scope open (beginAllocateToPool allows one per pool).
  void setPoolCaptureAllocator(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      CaptureId_t capture_id,
      std::shared_ptr<CCA::CUDAAllocator> allocator) override {
    TORCH_CHECK(allocator, "setPoolCaptureAllocator expects an allocator");
    std::lock_guard<std::mutex> lock(mutex_);
    const bool inserted = claims_
                              .try_emplace(
                                  PoolKey{device, mempool_id},
                                  Claim{std::move(allocator), capture_id})
                              .second;
    TORCH_INTERNAL_ASSERT(inserted, "a second claim on a CUDA graph pool");
    active_.store(claims_.size(), std::memory_order_relaxed);
  }
  // For a capture abandoned before capture_end (CUDAGraph::reset).
  void clearPoolCaptureAllocator(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      CaptureId_t capture_id) override {
    std::vector<std::shared_ptr<CCA::CUDAAllocator>> released;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = claims_.find({device, mempool_id});
    if (it != claims_.end() && it->second.capture == capture_id) {
      eraseLocked(it, released);
    }
  }
  size_t markCaptureEnd(c10::DeviceIndex device, CaptureId_t capture_id)
      override {
    if (C10_UNLIKELY(active_.load(std::memory_order_relaxed) != 0)) {
      std::vector<std::shared_ptr<CCA::CUDAAllocator>> released;
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = claims_.begin(); it != claims_.end();) {
        it = it->second.capture == capture_id ? eraseLocked(it, released)
                                              : std::next(it);
      }
    }
    return inner()->markCaptureEnd(device, capture_id);
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
  LAYERED_FORWARD(
      void,
      beginAllocateToPool,
      (c10::DeviceIndex device,
       MempoolId_t mempool_id,
       std::function<bool(cudaStream_t)> filter),
      (device, mempool_id, std::move(filter)))
  LAYERED_FORWARD(
      void,
      endAllocateToPool,
      (c10::DeviceIndex device, MempoolId_t mempool_id),
      (device, mempool_id))
  LAYERED_FORWARD(
      void,
      markCaptureBegin,
      (c10::DeviceIndex device,
       const c10::cuda::CUDAGraphMemory::CaptureRegistration& registration),
      (device, registration))
  LAYERED_FORWARD(void, markCaptureBegin, (c10::DeviceIndex device), (device))
  LAYERED_FORWARD(void, markCaptureEnd, (c10::DeviceIndex device), (device))
  LAYERED_FORWARD(bool, isCaptureContext, (c10::DeviceIndex device), (device))
  LAYERED_FORWARD(
      void,
      releasePool,
      (c10::DeviceIndex device, MempoolId_t mempool_id),
      (device, mempool_id))
  LAYERED_FORWARD(
      MempoolId_t,
      getAllocationPool,
      (c10::DeviceIndex device, cudaStream_t stream),
      (device, stream))
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
  // The capture allocator of the capture in progress into a pool.
  struct Claim {
    std::shared_ptr<CCA::CUDAAllocator> allocator;
    CaptureId_t capture;
  };
  struct Claimed {
    std::shared_ptr<CCA::CUDAAllocator> allocator;
    c10::DeviceIndex device;
    cudaStream_t stream;
  };

  CCA::CUDAAllocator* inner() const {
    return inner_.load(std::memory_order_relaxed);
  }

  // The erased claim's allocator goes to `released`, to be dropped after the
  // lock: dropping the last reference may run arbitrary (Python) code.
  std::map<PoolKey, Claim>::iterator eraseLocked(
      std::map<PoolKey, Claim>::iterator it,
      std::vector<std::shared_ptr<CCA::CUDAAllocator>>& released) {
    released.push_back(std::move(it->second.allocator));
    it = claims_.erase(it);
    active_.store(claims_.size(), std::memory_order_relaxed);
    return it;
  }

  // The capture allocator serving an allocation of `size` on the current
  // stream of the current device: inner would place it into the pool of a
  // capture with an allocator in progress. One relaxed load while no such
  // capture is in progress anywhere.
  std::optional<Claimed> claimant(size_t size) {
    if (C10_LIKELY(active_.load(std::memory_order_relaxed) == 0) || size == 0) {
      return std::nullopt;
    }
    const c10::DeviceIndex device = c10::cuda::current_device();
    const cudaStream_t s = c10::cuda::getCurrentCUDAStream(device).stream();
    // Only a capturing stream routes to a capture's pool: skip the caching
    // allocator's scope lookup (its lock and filters) for the others.
    if (c10::cuda::captureStatusMayInitCtx(s) ==
        c10::cuda::CaptureStatus::None) {
      return std::nullopt;
    }
    const MempoolId_t pool = inner()->getAllocationPool(device, s);
    if (pool == MempoolId_t{0, 0}) {
      return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = claims_.find({device, pool});
    if (it == claims_.end()) {
      return std::nullopt;
    }
    return Claimed{it->second.allocator, device, s};
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
        "CUDA out of memory: the CUDA graph capture allocator returned null "
        "for ",
        size,
        " bytes");
    if (C10_UNLIKELY(reinterpret_cast<uintptr_t>(ptr) % 256 != 0)) {
      c.allocator->raw_delete(ptr);
      TORCH_CHECK(
          false,
          "The CUDA graph capture allocator returned ",
          ptr,
          ", which is not aligned to 256 bytes");
    }
    return ptr;
  }

  std::atomic<CCA::CUDAAllocator*> inner_{nullptr};
  std::mutex mutex_;
  std::map<PoolKey, Claim> claims_;
  std::atomic<size_t> active_{0}; // claims_.size()
};

// Never destroyed: once enabled, it stays the current allocator until exit.
LayeredAllocator& layered() {
  static auto* allocator = new LayeredAllocator();
  return *allocator;
}

void deleteOwned(void* ctx) {
  std::unique_ptr<Owned> owned(static_cast<Owned*>(ctx));
  owned->allocator->raw_delete(owned->ptr);
}

} // namespace

void enableGraphAllocators() {
  auto& layer = layered();
  if (layer.installed()) {
    return;
  }
  static std::mutex enable_mutex;
  std::lock_guard<std::mutex> lock(enable_mutex);
  if (layer.installed()) {
    return;
  }
  auto* current = c10::cuda::CUDACachingAllocator::allocator.load();
  // Any time: the layer forwards to the caching allocator, which keeps owning
  // everything allocated so far.
  TORCH_CHECK(
      current->name() == "native",
      "CUDA graph allocators: the layer requires the native CUDA caching "
      "allocator, but the current allocator is '",
      current->name(),
      "'");
  layer.install(current);
}

bool graphAllocatorsEnabled() {
  return layered().installed();
}

} // namespace torch::cuda::CUDAPluggableAllocator

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
  auto* current = c10::cuda::CUDACachingAllocator::allocator.load();
  if (current_custom_allocator && current_custom_allocator.get() == current) {
    return current_custom_allocator;
  }
  // e.g. the native caching allocator, which is never destroyed: a handle
  // that does not own it, so that it can be made current again
  return std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
      current, [](c10::cuda::CUDACachingAllocator::CUDAAllocator*) {});
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

// Any time, including after the current allocator has been used. Memory stays
// with the allocator that made it (the caching allocator's is freed by it
// whichever allocator is current; a CUDAPluggableAllocator's through the
// current pluggable allocator, custom_raw_deleter, so it must not outlive a
// change away from it). Storages keep a pointer to the allocator that made
// them, so an allocator that has been current must stay alive. While it is
// current, the new allocator gets every allocator call in the process: after
// memory has been allocated, it should hand the calls it does not handle to
// the one it replaces. Only the calls the program actually makes need
// handling; the others can stay unsupported.
void changeCurrentAllocator(
    const std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>&
        allocator) {
  TORCH_CHECK(allocator, "change_current_allocator needs an allocator");
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

// The DataPtr context of a tensor made by the allocator a capture routes to.
struct Owned {
  std::shared_ptr<CCA::CUDAAllocator> allocator;
  void* ptr;
};

void deleteOwned(void* ctx) {
  std::unique_ptr<Owned> owned(static_cast<Owned*>(ctx));
  owned->allocator->raw_delete(owned->ptr);
}

// The allocator torch.cuda.memory._route_capture_allocations makes current,
// through change_current_allocator, while a CUDA graph capture runs. It sends
// the tensor allocations that the caching allocator (the allocator it
// replaces) would place into the capture's pool to the given allocator, one
// call per allocation at its exact size, and hands the other
// calls that occur during or after a capture to the caching allocator. The
// rest are CUDAPluggableAllocator's ("not supported" or nothing); add them
// here as they are needed.
class CaptureRoutingAllocator final : public CUDAPluggableAllocator {
 public:
  CaptureRoutingAllocator() : CUDAPluggableAllocator(nullptr, nullptr) {}

  void arm(std::shared_ptr<CCA::CUDAAllocator> allocator) {
    TORCH_CHECK(
        allocator,
        "routing a CUDA graph capture's allocations needs an allocator");
    const c10::DeviceIndex device = c10::cuda::current_device();
    const auto capture = c10::cuda::captureIdMayInitCtx(
        c10::cuda::getCurrentCUDAStream(device).stream());
    TORCH_CHECK(
        capture.has_value(),
        "a capture's allocations can only be routed inside a CUDA graph "
        "capture, on its capturing stream");
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(
        !route_, "a CUDA graph capture's allocations are already being routed");
    auto* current = CCA::allocator.load();
    TORCH_CHECK(
        current->name() == "native",
        "routing a CUDA graph capture's allocations requires the native CUDA caching "
        "allocator, but the current allocator is '",
        current->name(),
        "'");
    native_.store(current);
    route_ = Route{std::move(allocator), *capture, device};
    active_.store(true, std::memory_order_relaxed);
  }
  // Idempotent. Returns whether another capture began or ended while the
  // capture's allocations were routed.
  bool disarm() {
    std::optional<Route> released; // dropped after the lock: may run Python
    std::lock_guard<std::mutex> lock(mutex_);
    active_.store(false, std::memory_order_relaxed);
    released = std::move(route_);
    route_.reset();
    // a scope still open is older than the next routed capture
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
    return native()->allocate(size);
  }
  void recordStream(const c10::DataPtr& ptr, c10::cuda::CUDAStream stream)
      override {
    // A tensor of the allocator the capture routes to: that allocator handles
    // the stream use (a CUDAPluggableAllocator passes it to its
    // record_stream_fn, if set). After routing ends, the caching allocator
    // ignores memory it does not own.
    if (ptr.get_deleter() == &deleteOwned) {
      static_cast<Owned*>(ptr.get_context())
          ->allocator->recordStream(ptr, stream);
      return;
    }
    native()->recordStream(ptr, stream);
  }

  // Pool scopes opened while routing (use_mem_pool, NCCL and symmetric
  // memory pools) are newer than the capture's own scope, so they win over it,
  // as in the caching allocator's newest-first search: kept to answer the same.
  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)> filter) override {
    native()->beginAllocateToPool(device, mempool_id, filter);
    std::lock_guard<std::mutex> lock(mutex_);
    if (route_) {
      scopes_.push_back(Scope{device, mempool_id, std::move(filter)});
    }
  }
  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id)
      override {
    native()->endAllocateToPool(device, mempool_id);
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      if (it->device == device && it->pool == mempool_id) {
        scopes_.erase(std::next(it).base());
        break;
      }
    }
  }
  // These reach this allocator only when another capture (e.g. a conditional
  // node's body) begins or ends while it is current: CUDAGraphMemory sends the
  // registration-aware hooks only to the native allocator. Not supported
  // while routing: handed on, so that the caching allocator's count of
  // captures stays balanced, and reported when routing ends (raising here
  // would leave the capture half begun).
  void markCaptureBegin(c10::DeviceIndex device) override {
    other_capture_.store(true);
    native()->markCaptureBegin(device);
  }
  void markCaptureEnd(c10::DeviceIndex device) override {
    other_capture_.store(true);
    native()->markCaptureEnd(device);
  }

  // The caching allocator's, for the calls that occur during a capture: raw
  // allocations (and their deleter), MemPools made or released inside it,
  // device-to-device copies and peer access, statistics, and the backend name.
#define TO_NATIVE(ret, name, params, args) \
  ret name params override {               \
    return native()->name args;            \
  }
  TO_NATIVE(void*, raw_alloc, (size_t nbytes), (nbytes))
  TO_NATIVE(
      void*,
      raw_alloc_with_stream,
      (size_t nbytes, cudaStream_t stream),
      (nbytes, stream))
  TO_NATIVE(void, raw_delete, (void* ptr), (ptr))
  TO_NATIVE(c10::DeleterFnPtr, raw_deleter, () const, ())
  TO_NATIVE(
      void,
      createOrIncrefPool,
      (c10::DeviceIndex device,
       MempoolId_t mempool_id,
       std::shared_ptr<CCA::CUDAAllocator> allocator),
      (device, mempool_id, std::move(allocator)))
  TO_NATIVE(
      int,
      getPoolUseCount,
      (c10::DeviceIndex device, MempoolId_t mempool_id),
      (device, mempool_id))
  TO_NATIVE(
      void,
      releasePool,
      (c10::DeviceIndex device, MempoolId_t mempool_id),
      (device, mempool_id))
  TO_NATIVE(
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
  TO_NATIVE(
      void,
      enablePeerAccess,
      (c10::DeviceIndex dev, c10::DeviceIndex dev_to_access),
      (dev, dev_to_access))
  TO_NATIVE(
      c10::CachingDeviceAllocator::DeviceStats,
      getDeviceStats,
      (c10::DeviceIndex device),
      (device))
  TO_NATIVE(std::string, name, (), ())
#undef TO_NATIVE

 private:
  struct Route {
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

  CCA::CUDAAllocator* native() const {
    return native_.load(std::memory_order_relaxed);
  }

  // The routed-to allocator, if the caching allocator would place an
  // allocation of `size` on the current stream of the current device into the
  // pool of the routed capture: no newer pool scope takes it, and the stream
  // is capturing into that capture (the capture's own scope filter).
  std::optional<Claimed> claimant(size_t size) {
    if (C10_LIKELY(!active_.load(std::memory_order_relaxed)) || size == 0) {
      return std::nullopt;
    }
    const c10::DeviceIndex device = c10::cuda::current_device();
    const cudaStream_t s = c10::cuda::getCurrentCUDAStream(device).stream();
    std::lock_guard<std::mutex> lock(mutex_);
    if (!route_ || route_->device != device) {
      return std::nullopt;
    }
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      if (it->device == device && it->filter(s)) {
        return std::nullopt;
      }
    }
    const auto capture = c10::cuda::captureIdMayInitCtx(s);
    if (!capture.has_value() || *capture != route_->capture) {
      return std::nullopt;
    }
    return Claimed{route_->allocator, device, s};
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

  // the caching allocator, which this allocator replaces
  std::atomic<CCA::CUDAAllocator*> native_{nullptr};
  std::atomic<bool> active_{false};
  std::atomic<bool> other_capture_{false};
  std::mutex mutex_;
  std::optional<Route> route_;
  std::vector<Scope> scopes_;
};

// Never deleted: storages created while it is current keep a raw pointer to
// it (the resize rule dynamic_casts it, and storage ops like _lazy_clone call
// it). A handle that does not own it is what change_current_allocator gets.
CaptureRoutingAllocator& captureRouting() {
  static auto* allocator = new CaptureRoutingAllocator();
  return *allocator;
}

} // namespace

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
armCaptureRouting(
    std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator) {
  auto& routing = captureRouting();
  routing.arm(std::move(allocator));
  return std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
      &routing, [](c10::cuda::CUDACachingAllocator::CUDAAllocator*) {});
}

bool disarmCaptureRouting() {
  return captureRouting().disarm();
}

} // namespace torch::cuda::CUDAPluggableAllocator

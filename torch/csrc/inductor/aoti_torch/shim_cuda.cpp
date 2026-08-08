
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <shared_mutex>

namespace {
// No-op deleter: leaves the underlying graph-pool block alone so destroying a
// captured tensor handle does not free a block the captured graph still
// references. Such blocks are reclaimed wholesale via releasePool at teardown.
void noopDeleter(void*) {}

// Per-manager (per-AOTInductorModel) graph state: a PRIVATE graph mempool plus a
// single capture stream. Each AOTInductorModel instance owns one (created via
// aoti_torch_cuda_graph_pool_create, freed via ..._pool_destroy_handle), so
// concurrent instances in a model_container never share pool memory or a capture
// stream -- full isolation. Mirrors cudagraph_trees' per-tree-manager pool +
// self.stream. One capture stream per manager keeps a SINGLE cuBLAS workspace
// (keyed by (handle,stream)) for all of that manager's captures, avoiding
// per-node-stream workspace bloat; replay is stream-independent so this stream
// only affects recording.
struct AOTICudaGraphPool {
  c10::cuda::MempoolId_t pool;
  c10::cuda::CUDAStream stream;
  int32_t device_index;
  explicit AOTICudaGraphPool(int32_t dev)
      : pool(at::cuda::graph_pool_handle()),
        stream(c10::cuda::getStreamFromPool(false, dev)),
        device_index(dev) {}
};

// One captured graph, bound to its manager's pool (borrowed, not owned).
struct AOTICudaGraphContext {
  at::cuda::CUDAGraph graph;
  std::unique_ptr<c10::cuda::CUDAStreamGuard> stream_guard;
  AOTICudaGraphPool* pool;

  explicit AOTICudaGraphContext(AOTICudaGraphPool* p) : pool(p) {}
};

// Process-global READER-WRITER lock coordinating cuda-graph capture vs replay
// across all manager instances on a device. CAPTURE takes it EXCLUSIVE (write);
// REPLAY + output reconstruction take it SHARED (read). Two reasons concurrency
// here is unsafe: (1) the caching allocator's per-device capture bookkeeping
// (markCaptureBegin/beginAllocateToPool/...) and device-global capture state
// can't interleave across instances; (2) while one instance captures, another
// instance's replay must NOT run capture-unsafe CUDA queries -- output
// reconstruction calls getDeviceFromPtr (cudaPointerGetAttributes), which faults
// during a concurrent capture. So: at most one capture at a time AND no replay
// during a capture, but MANY replays run concurrently (shared). Capture is rare
// (warmup); steady-state serving only ever takes the shared lock, so concurrent
// inference across instances stays parallel.
std::shared_mutex& captureMutex() {
  static std::shared_mutex m;
  return m;
}
} // namespace

AOTITorchError aoti_torch_cuda_graph_pool_create(
    int32_t device_index,
    void** ret_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_handle = new AOTICudaGraphPool(device_index); });
}

AOTITorchError aoti_torch_cuda_graph_pool_destroy_handle(void* handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* p = reinterpret_cast<AOTICudaGraphPool*>(handle);
    // Drop this manager's capture-stream cuBLAS workspace BEFORE releasing the
    // pool (matching CUDAGraph::reset order): releasePool frees the pool segments
    // once use_count hits 0, so clearing the workspace afterwards would free an
    // already-freed block. Per-stream, so a concurrent manager is untouched.
    at::cuda::clearCublasWorkspacesForStream(p->stream.stream());
    c10::cuda::CUDACachingAllocator::releasePool(
        static_cast<c10::DeviceIndex>(p->device_index), p->pool);
    delete p;
  });
}

// Acquire/release the process-global capture/replay lock (see captureMutex). The
// tree runtime wraps record_node in capture_lock/unlock (EXCLUSIVE write) and
// wraps replay + output reconstruction in replay_lock/unlock (SHARED read).
AOTITorchError aoti_torch_cuda_graph_capture_lock() {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ captureMutex().lock(); });
}

AOTITorchError aoti_torch_cuda_graph_capture_unlock() {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ captureMutex().unlock(); });
}

AOTITorchError aoti_torch_cuda_graph_replay_lock() {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ captureMutex().lock_shared(); });
}

AOTITorchError aoti_torch_cuda_graph_replay_unlock() {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ captureMutex().unlock_shared(); });
}

AOTITorchError aoti_torch_create_cuda_guard(
    int32_t device_index,
    CUDAGuardHandle* ret_guard // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::cuda::CUDAGuard* guard = new at::cuda::CUDAGuard(device_index);
    *ret_guard = reinterpret_cast<CUDAGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_cuda_guard(CUDAGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::cuda::CUDAGuard*>(guard); });
}

AOTITorchError aoti_torch_cuda_guard_set_index(
    CUDAGuardHandle guard,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    reinterpret_cast<at::cuda::CUDAGuard*>(guard)->set_index(device_index);
  });
}

AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::cuda::CUDAStreamGuard* guard =
        new at::cuda::CUDAStreamGuard(at::cuda::getStreamFromExternal(
            static_cast<cudaStream_t>(stream), device_index));
    *ret_guard = reinterpret_cast<CUDAStreamGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::cuda::CUDAStreamGuard*>(guard); });
}

AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index,
    void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(cudaStream_t*)(ret_stream) = at::cuda::getCurrentCUDAStream(device_index);
  });
}

AOTITorchError aoti_torch_cuda_caching_allocator_raw_alloc(
    uint64_t nbytes,
    void** ret_ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (nbytes == 0) {
      *ret_ptr = nullptr;
      return AOTI_TORCH_SUCCESS;
    }

    *ret_ptr = c10::cuda::CUDACachingAllocator::raw_alloc(nbytes);

    if (*ret_ptr == nullptr) {
      TORCH_CHECK(
          false,
          "Failed to allocate ",
          nbytes,
          " bytes from CUDA caching allocator");
    }
  });
}

AOTITorchError aoti_torch_cuda_caching_allocator_raw_delete(void* ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (ptr != nullptr) {
      c10::cuda::CUDACachingAllocator::raw_delete(ptr);
    }
  });
}

AOTITorchError aoti_torch_cuda_graph_create(
    AOTICudaGraphHandle* ret_handle,
    void* pool_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* ctx = new AOTICudaGraphContext(
        reinterpret_cast<AOTICudaGraphPool*>(pool_handle));
    *ret_handle = reinterpret_cast<AOTICudaGraphHandle>(ctx);
  });
}

AOTITorchError aoti_torch_cuda_graph_begin_capture(
    AOTICudaGraphHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* ctx = reinterpret_cast<AOTICudaGraphContext*>(handle);
    auto stream = ctx->pool->stream;
    stream.synchronize();
    ctx->stream_guard = std::make_unique<c10::cuda::CUDAStreamGuard>(stream);
    // ThreadLocal (not the default Global) capture mode: serialized against other
    // captures by the global capture lock, this avoids disrupting other model
    // instances' concurrent replays on other threads.
    ctx->graph.capture_begin(ctx->pool->pool, cudaStreamCaptureModeThreadLocal);
  });
}

AOTITorchError aoti_torch_cuda_graph_end_capture(
    AOTICudaGraphHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* ctx = reinterpret_cast<AOTICudaGraphContext*>(handle);
    ctx->graph.capture_end();
    ctx->stream_guard.reset();
  });
}

AOTITorchError aoti_torch_cuda_graph_replay(AOTICudaGraphHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* ctx = reinterpret_cast<AOTICudaGraphContext*>(handle);
    ctx->graph.replay();
  });
}

AOTITorchError aoti_torch_cuda_graph_get_stream(
    AOTICudaGraphHandle handle,
    void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* ctx = reinterpret_cast<AOTICudaGraphContext*>(handle);
    *(cudaStream_t*)(ret_stream) = ctx->pool->stream.stream();
  });
}

AOTITorchError aoti_torch_cuda_graph_destroy(AOTICudaGraphHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    delete reinterpret_cast<AOTICudaGraphContext*>(handle);
  });
}

AOTITorchError aoti_torch_cuda_graph_pool_ensure_created(void* pool_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // Materialize this manager's PrivatePool (empty) so releasePool at teardown
    // has a real pool to release even when no partition is ever captured, and so
    // the constructor's matching pool use ref balances the destructor's drop. A
    // no-op begin/end allocation scope creates the pool via create_or_incref_pool
    // without allocating anything. This adds one pool use ref (only raises
    // use_count, never drives it negative), so the pool may stay reserved until
    // process exit; harmless for inference.
    auto* p = reinterpret_cast<AOTICudaGraphPool*>(pool_handle);
    auto dev = static_cast<c10::DeviceIndex>(p->device_index);
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        dev, p->pool, [](cudaStream_t) { return false; });
    c10::cuda::CUDACachingAllocator::endAllocateToPool(dev, p->pool);
  });
}

AOTITorchError aoti_torch_cuda_graph_sync_before_record(int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // Mirror cudagraph_trees' pre-record torch.cuda.synchronize(). The single
    // shared capture stream means every partition reuses one cuBLAS workspace
    // slot; without ordering, a previous partition's first-replay on the caller's
    // stream and this partition's warmup/capture on the capture stream would race
    // on that workspace. A full device sync serializes all prior work before we
    // touch the pool or capture, which subsumes ordering the capture stream after
    // the caller's stream (everything is idle afterward). Recording-only.
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_index));
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  });
}

AOTITorchError aoti_torch_cuda_graph_clear_cublas_workspaces(void* pool_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // Drop this manager's capture-stream cuBLAS workspace after capture. cuBLAS
    // allocates its matmul workspace through the caching allocator, so during
    // capture it lands in the graph pool and is tracked in a per-(handle,stream)
    // global map. Clearing this stream's entry stops the map from referencing a
    // block that teardown frees, which would double-free at teardown
    // (CUDAGraph::reset -> clearCublasWorkspacesForStream). Per-stream (not the
    // global clear) so a concurrent manager's workspace is left intact. Mirrors
    // cudagraph_trees' clear_cublas_manager (clear after each recording).
    auto* p = reinterpret_cast<AOTICudaGraphPool*>(pool_handle);
    at::cuda::clearCublasWorkspacesForStream(p->stream.stream());
  });
}

AOTITorchError aoti_torch_cuda_graph_device_used_bytes(
    int32_t device_index,
    int64_t* ret_bytes) {
  // Debug/instrumentation only: total GPU bytes in use on the device
  // (total - free). On an otherwise-idle benchmark GPU this reflects this
  // process's reservation, so the delta across recorded shapes isolates the
  // graph-pool growth (the cross-shape memory-sharing signal).
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::cuda::CUDAGuard guard(device_index);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    *ret_bytes = static_cast<int64_t>(total_bytes - free_bytes);
  });
}

AOTITorchError aoti_torch_storage_set_noop_deleter(AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = reinterpret_cast<at::Tensor*>(tensor);
    c10::DataPtr& data_ptr =
        t->storage().unsafeGetStorageImpl()->mutable_data_ptr();
    // expected == current deleter, so the swap always succeeds; ignore result.
    (void)data_ptr.compare_exchange_deleter(data_ptr.get_deleter(), &noopDeleter);
  });
}

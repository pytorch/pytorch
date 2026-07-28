#include <ATen/core/CachingHostAllocator.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/MemPool.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAFunctions.h>

#include <cstddef>
#include <optional>

namespace at::cuda {

// To support stream capture across multiple threads, we use a global
// hashmap mapping cuda stream capture IDs to CUDAGraph objects. This
// was originally a thread_local std::stack<CUDAGraph*>, but that was
// not acceptable since stream capture does span threads in certain
// circumstances (in particular, during autograd).
static std::mutex _currently_capturing_graphs_mutex;
static ska::flat_hash_map<CaptureId_t, CUDAGraph*> _currently_capturing_graphs;

#if defined(USE_ROCM)
// Returns true when at least one CUDAGraph capture is currently active in this
// process. Uses the same mutex-protected capture map as capture lifecycle
// bookkeeping.
bool is_graph_capture_active() {
  std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
  return !_currently_capturing_graphs.empty();
}
#endif // defined(USE_ROCM)

CUDAGraph* get_graph_from_capture_id(CaptureId_t capture_id) {
  std::lock_guard<std::mutex> lock(_currently_capturing_graphs_mutex);
  auto it = _currently_capturing_graphs.find(capture_id);
  if (it != _currently_capturing_graphs.end()) {
    return it->second;
  }
  return nullptr;
}

MempoolId_t graph_pool_handle() {
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  return at::cuda::MemPool::graph_pool_handle();
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in CUDAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * describes memory management for captures.
 */

CUDAGraph::CUDAGraph(bool keep_graph)
  // CUDAStreams may not be default-constructed.
  : capture_stream_(at::cuda::getCurrentCUDAStream()),
    keep_graph_(keep_graph) {
}

void CUDAGraph::register_generator_state(
    c10::intrusive_ptr<at::CUDAGeneratorState> state) {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  TORCH_CHECK(
      active_segment_id_.load() == capture_id_,
      "RNG within a segmented CUDA graph capture is not supported yet.");
#endif
  captured_generator_states_[std::move(state)] = 0;
}

bool CUDAGraph::has_retained_pool(MempoolId_t pool) const {
  for (const auto& retained_pool : retained_mempool_ids_) {
    if (retained_pool == pool) {
      return true;
    }
  }
  return false;
}

void CUDAGraph::record_retained_pool(MempoolId_t pool) {
  if (!has_retained_pool(pool)) {
    retained_mempool_ids_.push_back(pool);
  }
}

void CUDAGraph::retain_pool(MempoolId_t pool) {
  TORCH_CHECK(
      capture_id_ != 0 && !capture_ended_,
      "CUDAGraph::retain_pool may only be called during capture.");
  TORCH_CHECK(
      pool.first != 0 || pool.second != 0,
      "CUDAGraph::retain_pool expected a non-default memory pool.");
  if (has_retained_pool(pool)) {
    return;
  }
  c10::cuda::CUDACachingAllocator::createOrIncrefPool(capture_dev_, pool);
  record_retained_pool(pool);
}

template <>
std::function<bool(cudaStream_t)> CUDAGraph::create_allocate_filter<cudaStream_t>() const {
  return [this](cudaStream_t stream) {
    auto capture_id_opt = c10::cuda::captureIdMayInitCtx(stream);
    return capture_id_opt.has_value() &&
        capture_id_opt.value() == active_segment_id_.load();
  };
}

template <>
std::function<bool(c10::Stream)> CUDAGraph::create_allocate_filter<c10::Stream>() const {
  return [this](c10::Stream stream) {
    cudaStream_t cuda_stream = CUDAStream(CUDAStream::UNCHECKED, stream);
    auto capture_id_opt = c10::cuda::captureIdMayInitCtx(cuda_stream);
    return capture_id_opt.has_value() &&
        capture_id_opt.value() == active_segment_id_.load();
  };
}

void CUDAGraph::register_active_segment(CaptureId_t capture_id) {
  std::lock_guard<std::mutex> lock(_currently_capturing_graphs_mutex);
  _currently_capturing_graphs.emplace(capture_id, this);
  active_segment_id_.store(capture_id);
}

void CUDAGraph::capture_begin(MempoolId_t pool/*={0,0}*/, cudaStreamCaptureMode capture_mode) {
  TORCH_CHECK(graph_exec_ == nullptr,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  capture_mode_ = capture_mode;

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_dev_ = c10::cuda::current_device();

#if defined(USE_ROCM)
  // hipBLASLt handles are per-(device, stream) on ROCm and lazily created.
  // Ensure the handle for the intended capture stream exists before
  // capture begins, because hipblasLtCreate performs internal allocations
  // that are not allowed once stream capture is active.
  if (at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublaslt) {
    (void)at::cuda::getCurrentCUDABlasLtHandle();
  }
#endif

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Create graph pool handle using is_user_created=false.
    // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
    mempool_id_ = at::cuda::MemPool::graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
  // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
  // due to the capture status being updated _after_ a capture had already started.
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      capture_dev_, mempool_id_, create_allocate_filter<cudaStream_t>());
  record_retained_pool(mempool_id_);
  at::getHostAllocator(at::kCUDA)->begin_allocate_to_pool(
      mempool_id_, create_allocate_filter<c10::Stream>());
  pool_acquired_ = true;
  pool_routing_ = true;

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));
  c10::cuda::CUDACachingAllocator::markCaptureBegin(capture_dev_);

  auto capture_id_opt = c10::cuda::captureIdMayInitCtx(stream);
  TORCH_INTERNAL_ASSERT(
      capture_id_opt.has_value(),
      "Stream should be actively capturing after cudaStreamBeginCapture");
  capture_id_ = capture_id_opt.value();
  register_active_segment(capture_id_);
}

// capture_end is split so callers can run work on the captured cudaGraph_t
// (e.g. read its id, dump it, transform it) in the window between the end of
// capture and finalization, when graph_ is live for both keep_graph modes.
// capture_end_post finalizes by destroying the template for keep_graph=false;
// instantiation is driven separately (by capture_end for C++ callers, or by the
// Python wrapper) so it has a single entry point. capture_end runs the whole
// sequence for callers that don't need the window.
cudaError_t CUDAGraph::end_active_segment(cudaGraph_t* graph) {
  CaptureId_t active_segment_id = active_segment_id_.load();
  TORCH_CHECK(
      active_segment_id != 0,
      "capture_end() called before capture_begin().");
  bool was_registered = false;
  {
    std::lock_guard<std::mutex> lock(_currently_capturing_graphs_mutex);
    was_registered = _currently_capturing_graphs.erase(active_segment_id) == 1;
  }

  cudaError_t error = cudaStreamEndCapture(capture_stream_, graph);
  c10::cuda::CUDACachingAllocator::markCaptureEnd(capture_dev_);
  active_segment_id_.store(0);
  TORCH_INTERNAL_ASSERT(
      was_registered, "Active CUDA graph capture segment was not registered.");
  return error;
}

void CUDAGraph::capture_end_pre() {
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream.stream() == capture_stream_.stream(),
              "Capture must end on the same stream it began on.");
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  TORCH_CHECK(
      suspended_capture_frames_.empty(),
      "All conditional CUDA graph bodies must be ended before capture_end().");
#endif

  // Capture is over once cudaStreamEndCapture returns (success or failure).
  // Clear bookkeeping before propagating the return status so watchdog-side
  // checks cannot observe stale "capture active" state on error paths.
  cudaGraph_t captured_graph = nullptr;
  cudaError_t endCaptureErr = end_active_segment(&captured_graph);
  if (endCaptureErr == cudaSuccess && graph_ == nullptr) {
    graph_ = captured_graph;
  }
  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      capture_dev_, mempool_id_);
  at::getHostAllocator(at::kCUDA)->end_allocate_to_pool(mempool_id_);
  pool_routing_ = false;
  AT_CUDA_CHECK(endCaptureErr);

  TORCH_CHECK(captured_graph != nullptr, "Invalid capture.");
  TORCH_CHECK(
      captured_graph == graph_,
      "The final CUDA capture segment returned a different root graph.");

  for (auto& [generator_state, wholegraph_increment] :
       captured_generator_states_) {
    wholegraph_increment = generator_state->capture_epilogue(capture_id_);
  }

  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
      TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
                 "attempted to be captured on wrong device or stream.");
  }

  capture_ended_ = true;
}

void CUDAGraph::capture_end_post() {
  // Destroy-only: when keep_graph=false the template is not retained. The graph
  // must already be instantiated (capture_end and the Python wrapper instantiate
  // before calling this).
  if (!keep_graph_ && graph_ != nullptr) {
    AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    graph_ = nullptr;
  }
}

void CUDAGraph::capture_end() {
  capture_end_pre();
  if (!keep_graph_) {
    instantiate();
  }
  capture_end_post();
}

void CUDAGraph::instantiate() {
  TORCH_CHECK(capture_ended_, "capture_end() must have been called before calling instantiate");

  if (graph_exec_ != nullptr) {
    TORCH_CHECK(keep_graph_, "instantiate() is intended to be called by the user only when keep_graph=true");
    AT_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
    graph_exec_ = nullptr;
  }
  // In typical graph usage some tensors (e.g. the tensors used for graph IO) are not freed
  // between replays.
  // If Pytorch compiles and runs with a CUDA 11.4+ toolkit, there's a chance the allocator backend
  // is cudaMallocAsync.
  // cudaMallocAsync is generally graph-safe, but if some tensors are not freed between replays,
  // the graph's internal bookkeeping requires that we instantiate with
  // cudaGraphInstantiateFlagAutoFreeOnLaunch. See
  // cudaGraphLaunch
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597
  // cudaGraphInstantiateWithFlags
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1ga2c652a24ba93e52b99a47bec0888233
#if !defined(USE_ROCM)
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch | cudaGraphInstantiateFlagUseNodePriority));
#else
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch));
#endif
}

void CUDAGraph::replay() {
  TORCH_CHECK(capture_ended_,
              "Called CUDAGraph::replay without a preceding successful capture.");
  // Instantiating on demand is handled by the Python replay() wrapper (which
  // can do so for keep_graph=true). At this level the exec graph must exist.
  TORCH_CHECK(graph_exec_ != nullptr,
              "Called CUDAGraph::replay before the graph was instantiated; "
              "call instantiate() first.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increment] :
       captured_generator_states_) {
    generator_state->replay_prologue(capture_id_, wholegraph_increment);
  }
  // graph_exec_ may be replayed in any stream.
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));
}

void CUDAGraph::enable_debug_mode() {
  // Debug mode just retains the template after capture so it can be inspected
  // (e.g. dumped); that is exactly what keep_graph does. Unify on keep_graph_
  // rather than a second flag. dot dumping itself lives in Python now
  // (torch.cuda.CUDAGraph.debug_dump via cuda.bindings).
  keep_graph_ = true;
}

cudaGraph_t CUDAGraph::raw_cuda_graph() {
  TORCH_CHECK(capture_ended_ && graph_ != nullptr,
      "No cudaGraph_t is available: either capture_end() has not been called, "
      "or the underlying cudaGraph_t was destroyed (keep_graph=false, and "
      "capture has been finalized).");
  return graph_;
}

cudaGraphExec_t CUDAGraph::raw_cuda_graph_exec() {
  TORCH_CHECK(
      graph_exec_ != nullptr,
      "You cannot access the raw cudaGraphExec_t instance until instantiate() has been called");
  return graph_exec_;
}

void CUDAGraph::reset() {
  // These checks warn instead of throwing: reset() is called from the
  // destructor, and at least one CI build refuses to compile with a throwing
  // destructor. Resource cleanup lives here in C++ so it runs on garbage
  // collection regardless of Python state; the Python __del__ on
  // torch.cuda.CUDAGraph only tears down bookkeeping and intentionally does not
  // call reset().
  //
  // If capture_begin, the capture, or capture_end failed at some point, this CUDAGraph, the generator,
  // and the allocator could end up in all kinds of weird states depending where failure occurred.
  // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
  // a Jupyter notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.

  // See Note [RNG state tensor lifetime and recordStream] in
  // CUDAGeneratorImpl.cpp — recordStream in setup_for_replay ensures the
  // allocator won't recycle these tensors until in-flight replays finish.
  if (capture_id_ != 0) {
    for (auto& [generator_state, wholegraph_increment] : captured_generator_states_) {
      generator_state->remove_capture_state(capture_id_);
    }
  }
  captured_generator_states_.clear();

  if (active_segment_id_.load() != 0) {
    cudaGraph_t abandoned_graph = nullptr;
    cudaError_t end_capture_error =
        end_active_segment(&abandoned_graph);
    C10_CUDA_CHECK_WARN(end_capture_error);
    if (end_capture_error == cudaSuccess && abandoned_graph != nullptr &&
        graph_ == nullptr) {
      graph_ = abandoned_graph;
    }
  }
  capture_id_ = 0;

  if (pool_acquired_) {
    if (pool_routing_) {
      // Capture was abandoned before capture_end() ran, so the allocator is
      // still routing allocations to this pool. Stop that before releasing so
      // the pool is left in a consistent, freeable state.
      c10::cuda::CUDACachingAllocator::endAllocateToPool(
          capture_dev_, mempool_id_);
      at::getHostAllocator(at::kCUDA)->end_allocate_to_pool(mempool_id_);
      pool_routing_ = false;
    }

    // Clean up cuBLAS workspaces allocated on the capture stream, otherwise live allocations prevent
    // private pool cleanup
    clearCublasWorkspacesForStream(capture_stream_.stream());

    // notifyCaptureDestroy may throw. How should we handle this?
    for (const auto& pool : retained_mempool_ids_) {
      c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, pool);
    }
    at::getHostAllocator(at::kCUDA)->release_pool(mempool_id_);
    pool_acquired_ = false;
  }
  retained_mempool_ids_.clear();
  capture_ended_ = false;
  if (graph_ != nullptr) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
    graph_ = nullptr;
  }
  if (graph_exec_ != nullptr) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    graph_exec_ = nullptr;
  }
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  suspended_capture_frames_.clear();
#endif
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t CUDAGraph::pool() {
  TORCH_CHECK(capture_ended_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

std::vector<MempoolId_t> CUDAGraph::pools() {
  TORCH_CHECK(capture_ended_,
              "Called CUDAGraph::pools() without a preceding successful capture.");
  return retained_mempool_ids_;
}

CUDAGraph::~CUDAGraph() {
  reset();

// There are recent HIP changes where hipGraphExecDestroy doesn't immediately free memory.
// They wait for next sync point in order to free the memory, this is to ensure that all
// hipGraphLaunch are finished before we release any memory. This feature was enabled in rocm6.2.
// We need to ensure all async operations finish before deleting the object.
#if defined(USE_ROCM)
  if (capture_dev_ != UNDEFINED_DEVICE) // check if capture_dev_ contains the real device id
  {
    AT_CUDA_CHECK(cudaSetDevice(capture_dev_));
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif
}

CUDAGraph* CUDAGraph::get_currently_capturing_graph() {
  std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
  auto capture_id_opt = c10::cuda::currentStreamCaptureIdMayInitCtx();
  TORCH_CHECK(
      capture_id_opt.has_value(),
      "The current stream is not currently capturing.");
  TORCH_CHECK(
      _currently_capturing_graphs.count(capture_id_opt.value()),
      "get_currently_capturing_graph() can be used only between capture_begin() and capture_end(). Did you use a stream without making it depend upon the original stream used for capture?");
  return _currently_capturing_graphs.at(capture_id_opt.value());
}

void CUDAGraph::begin_capture_to_if_node(
    const at::Tensor& scalar_cuda_pred_tensor) {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  begin_capture_to_conditional_node(
      scalar_cuda_pred_tensor, cudaGraphCondTypeIf);
#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
  return;
#endif
}

void CUDAGraph::begin_capture_to_while_node(
    const at::Tensor& scalar_cuda_pred_tensor) {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  begin_capture_to_conditional_node(
      scalar_cuda_pred_tensor, cudaGraphCondTypeWhile);
#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
  return;
#endif
}

#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
void CUDAGraph::begin_segment(
    cudaGraph_t graph,
    cudaGraphNode_t dependency) {
  TORCH_INTERNAL_ASSERT(graph != nullptr);

  AT_CUDA_CHECK(cudaStreamBeginCaptureToGraph(
      capture_stream_,
      graph,
      dependency == nullptr ? nullptr : &dependency,
      nullptr,
      dependency == nullptr ? 0 : 1,
      capture_mode_));
  c10::cuda::CUDACachingAllocator::markCaptureBegin(capture_dev_);

  auto capture_info = c10::cuda::captureInfoMayInitCtx(capture_stream_);
  TORCH_INTERNAL_ASSERT(
      capture_info.status == c10::cuda::CaptureStatus::Active,
      "Stream should be capturing after cudaStreamBeginCaptureToGraph");
  TORCH_INTERNAL_ASSERT(capture_info.graph == graph);
  register_active_segment(capture_info.id);
}

void CUDAGraph::begin_capture_to_conditional_node(
    const at::Tensor& scalar_cuda_pred_tensor,
    cudaGraphConditionalNodeType conditional_type) {
  TORCH_CHECK(
      graph_exec_ == nullptr,
      "This CUDAGraph instance already owns a captured graph.");

  TORCH_CHECK(
      c10::cuda::CUDACachingAllocator::name() == "native",
      "Segmented CUDA graph capture requires the native caching allocator.");
  TORCH_CHECK(
      !c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::graph_capture_record_stream_reuse(),
      "'graph_capture_record_stream_reuse:True' allocator config does not work with conditional control flow in a cuda graph today. See issue #175001 for updates");
  TORCH_CHECK(
      captured_generator_states_.empty(),
      "RNG within a segmented CUDA graph capture is not supported yet.");
  TORCH_CHECK(
      getCurrentCUDAStream().stream() == capture_stream_.stream(),
      "Conditional nodes must be captured on the root CUDA graph stream.");

  auto capture_info = c10::cuda::captureInfoMayInitCtx(capture_stream_);
  TORCH_CHECK(
      capture_info.status == c10::cuda::CaptureStatus::Active,
      "capture_begin() must be called before begin_capture_to_conditional_node()");
  cudaGraph_t currently_capturing_graph = capture_info.graph;
  try {
    cudaGraphConditionalHandle handle{};
    AT_CUDA_CHECK(cudaGraphConditionalHandleCreate(
        &handle, currently_capturing_graph, 0, 0));
    set_conditional_handle(handle, scalar_cuda_pred_tensor);

    const cudaGraphNode_t* dependencies = nullptr;
    const cudaGraphEdgeData* edge_data = nullptr;
    size_t dependency_count = 0;
    cudaStreamCaptureStatus status{};
#if CUDA_VERSION >= 13000
    AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
        capture_stream_,
        &status,
        nullptr,
        nullptr,
        &dependencies,
        &edge_data,
        &dependency_count));
#else
    AT_CUDA_CHECK(cudaStreamGetCaptureInfo_v3(
        capture_stream_,
        &status,
        nullptr,
        nullptr,
        &dependencies,
        &edge_data,
        &dependency_count));
#endif
    TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatusActive);

    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = conditional_type;
    params.conditional.size = 1;

    cudaGraphNode_t conditional_node{};
#if CUDA_VERSION >= 13000
    AT_CUDA_CHECK(cudaGraphAddNode(
        &conditional_node,
        currently_capturing_graph,
        dependencies,
        edge_data,
        dependency_count,
        &params));
#else
    AT_CUDA_CHECK(cudaGraphAddNode_v2(
        &conditional_node,
        currently_capturing_graph,
        dependencies,
        edge_data,
        dependency_count,
        &params));
#endif

    cudaGraph_t completed_graph = nullptr;
    cudaError_t error = end_active_segment(&completed_graph);
    if (error == cudaSuccess && graph_ == nullptr) {
      graph_ = completed_graph;
    }
    AT_CUDA_CHECK(error);
    TORCH_CHECK(
        completed_graph == currently_capturing_graph,
        "A CUDA capture segment returned a different graph than its target.");

    cudaGraph_t conditional_body = params.conditional.phGraph_out[0];
    suspended_capture_frames_.push_back(
        {currently_capturing_graph, conditional_node, handle});
    begin_segment(conditional_body);
  } catch (...) {
    reset();
    throw;
  }
}
#endif // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)

void CUDAGraph::end_capture_to_conditional_node() {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  TORCH_CHECK(
      !suspended_capture_frames_.empty(),
      "Missing suspended capture frame for conditional node.");
  TORCH_CHECK(
      getCurrentCUDAStream().stream() == capture_stream_.stream(),
      "Conditional nodes must be captured on the root CUDA graph stream.");

  try {
    cudaGraph_t completed_graph = nullptr;
    AT_CUDA_CHECK(end_active_segment(&completed_graph));
    TORCH_CHECK(
        completed_graph != nullptr,
        "A conditional CUDA graph body returned an invalid graph.");
    SuspendedCaptureFrame parent = suspended_capture_frames_.back();
    suspended_capture_frames_.pop_back();
    begin_segment(parent.graph, parent.resume_after);
  } catch (...) {
    reset();
    throw;
  }

#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
#endif
}

void CUDAGraph::set_conditional_handle_for_current_node(
    const at::Tensor& scalar_cuda_pred_tensor) {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  TORCH_INTERNAL_ASSERT(
      !suspended_capture_frames_.empty(),
      "No active CUDA graph conditional node.");
  set_conditional_handle(
      suspended_capture_frames_.back().handle, scalar_cuda_pred_tensor);
#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
#endif
}

} // namespace at::cuda

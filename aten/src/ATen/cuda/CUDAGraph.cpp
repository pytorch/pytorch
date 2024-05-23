#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

#include <chrono>
#include <cstddef>
#include <thread>

namespace at::cuda {

void external_stream_deleter(cudaStream_t* stream) {
  if (stream != nullptr) {
    cudaStreamDestroy(*stream);
    delete stream;
  }
}

namespace {
UniquePtrExternalCudaStream create_external_stream() {
  // From:
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3
  // "Capture must be ended on the same stream in which it was initiated, and it
  // may only be initiated if the stream is not already in capture mode."

  // Since pytorch uses a pool of 32 pre-allocated cuda streams,
  // should a user nest 32 conditional nodes, there would be an error
  // for the 32nd node, since that node's stream would already be in
  // capture mode. The easiest solution is to handle stream creation
  // and deletion ourselves.

  // we use cudaStreamNonBlocking because every default cuda stream in
  // pytorch uses that flag for all streams used for stream capture
  // (see kDefaultFlags in CUDAStream.cpp). This would need to be kept
  // in sync, should that ever change. Or kDefaultFlags needs to be
  // exposed in a header file.
  auto stream_ptr = std::make_unique<cudaStream_t>();
  AT_CUDA_CHECK(
      cudaStreamCreateWithFlags(stream_ptr.get(), cudaStreamNonBlocking));
  return UniquePtrExternalCudaStream(
      stream_ptr.release(), external_stream_deleter);
}
} // anonymous namespace

static bool _cuda_graphs_debug = false;
constexpr int kSynchronizeBusyWaitMillis = 10;

// To support stream capture across multiple threads, we use a global
// hashmap mapping cuda stream capture IDs to CUDAGraph objects. This
// was originally a thread_local std::stack<CUDAGraph*>, but that was
// not acceptable since stream capture does span threads in certain
// circumstances (in particular, during autograd).
static std::mutex _currently_capturing_graphs_mutex;
static ska::flat_hash_map<CaptureId_t, CUDAGraph*> _currently_capturing_graphs;

MempoolId_t graph_pool_handle() {
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  return c10::cuda::MemPool::graph_pool_handle();
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

std::atomic<int> CUDAGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a NCCL watchdog so that they
// can be resolved before the capture begins. Note that event queries are not allowed during a
// graph capture in the default capture mode.
void CUDAGraph::inc_pending_event_queries() {
  pending_event_queries++;
}

void CUDAGraph::dec_pending_event_queries() {
  TORCH_INTERNAL_ASSERT(pending_event_queries > 0,
    "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
  pending_event_queries--;
}

int CUDAGraph::num_pending_event_queries() {
  return pending_event_queries;
}

CUDAGraph::CUDAGraph()
  // CUDAStreams may not be default-constructed.
  : capture_stream_(at::cuda::getCurrentCUDAStream()) {
}

void CUDAGraph::register_generator_state(
    c10::intrusive_ptr<at::CUDAGeneratorState> state) {
  captured_generator_states_[std::move(state)] = 0;
}

void CUDAGraph::register_generator_state(const at::Generator& generator) {
  c10::intrusive_ptr<CUDAGeneratorImpl> cuda_gen =
      dynamic_intrusive_pointer_cast<CUDAGeneratorImpl>(
          generator.getIntrusivePtr());
  cuda_gen->register_graph(this);
}

void CUDAGraph::capture_begin(MempoolId_t pool/*={0,0}*/, cudaStreamCaptureMode capture_mode) {
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  capture_mode_ = capture_mode;

  // default generator is always registered
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      std::nullopt, cuda::detail::getDefaultCUDAGenerator());
  gen->register_graph(this);

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->capture_prologue();
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_dev_ = c10::cuda::current_device();

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
    mempool_id_ = c10::cuda::MemPool::graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
  // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
  // due to the capture status being updated _after_ a capture had already started.
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, create_allocate_filter());

  // At this point, any NCCL watchdogs should be aware that we are in capture mode
  // and therefore should not enqueue any additional work that could be event-queried.
  // We still must wait on any existing work that has not been cleaned up.
  while (num_pending_event_queries()) {
    TORCH_WARN_ONCE("Waiting for pending NCCL work to finish before starting graph capture.");
    std::this_thread::sleep_for(
      std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));

  cudaStreamCaptureStatus status{};
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &capture_id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  {
    std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
    _currently_capturing_graphs.emplace(capture_id_, this);
  }
}

void CUDAGraph::capture_end() {
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));

  {
    std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
    TORCH_CHECK(
        _currently_capturing_graphs.count(capture_id_),
        "capture_end() called before capture_begin().");
    _currently_capturing_graphs.erase(capture_id_);
  }

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");
  has_graph_ = true;

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
#if ((defined(CUDA_VERSION) && CUDA_VERSION >= 11040) || (defined(USE_ROCM) && ROCM_VERSION >= 60200))
  int version = 0;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
    // Trailing NULL, NULL, 0 arguments were recommended by Cuda driver people,
    // who prefer not to report error message through these arguments moving forward
    // (they prefer return value, or errors on api calls internal to the capture)
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
#else
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
#endif
//Since ROCm 6.2, we want to go down this path as hipGraphExecDestroy in the destructor will not immediately free the memory.
//It will wait for the next sync operation. cudaGraphInstantiateFlagAutoFreeOnLaunch will add async frees after graph launch.
#if ((defined(CUDA_VERSION) && CUDA_VERSION >= 11040) || (defined(USE_ROCM) && ROCM_VERSION >= 60200))
  } else {
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch));
  }
#endif

  has_graph_exec_ = true;

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
      TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
                 "attempted to be captured on wrong device or stream.");
  }

  // check if debug path is set
  if (!_cuda_graphs_debug) {
    // Now that we've instantiated graph_ into graph_exec_,
    // we don't need graph_ anymore.
    AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    has_graph_ = false;
  } else {
    TORCH_WARN("DEBUG: TORCH_CUDAGRAPHS_DEBUG_PATH detected. graph_ will not be freed until debug_dump is called.");
  }
}

void CUDAGraph::replay() {
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }
  // graph_exec_ may be replayed in any stream.
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  int version = 0;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    // Workaround for bug in libcuda.so that causes replayed graphs with
    // certain topologies to be corrupted (kernels elided, internal syncs
    // ignored) when replayed back to back without a sync in between.
    // The bug is fixed in CUDA 11.4+.
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void CUDAGraph::enable_debug_mode() {
  _cuda_graphs_debug = true;
}

void CUDAGraph::debug_dump(const std::string& debug_path) {
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11030)|| defined(USE_ROCM)
  if (_cuda_graphs_debug) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling cudaGraphDebugDotPrint() with ", debug_path);
      C10_CUDA_CHECK_WARN(cudaGraphDebugDotPrint(graph_, debug_path.c_str(), cudaGraphDebugDotFlagsVerbose)); // most verbose output
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
      has_graph_ = false;
    }
  } else {
    TORCH_WARN("CUDA Graphs debug not enabled, set with [graph].enable_debug_mode()");
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.3 or ROCM >= 5.6");
#endif
}

void CUDAGraph::reset() {
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this CUDAGraph, the generator,
  // and the allocator could end up in all kinds of weird states depending where failure occurred.
  // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
  // a Jupyter notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.
  if (has_graph_ || has_graph_exec_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t CUDAGraph::pool() {
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

CUDAGraph::~CUDAGraph() {
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->unregister_graph(this);
  }
  reset();

// There are recent HIP changes where hipGraphExecDestroy doesn't immediately free memory.
// They wait for next sync point in order to free the memory, this is to ensure that all
// hipGraphLaunch are finished before we release any memory. This feature was enabled in rocm6.2.
// We need to ensure all async opreations finish before deleting the object.
#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
  if (capture_dev_ != UNDEFINED_DEVICE) // check if capture_dev_ contains the real device id
  {
    AT_CUDA_CHECK(cudaSetDevice(capture_dev_));
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif
}

CUDAGraph* CUDAGraph::get_currently_capturing_graph() {
  std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
  cudaStreamCaptureStatus status{};
  CaptureId_t current_capture_id = -1;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &current_capture_id));
  TORCH_CHECK(
      status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive,
      "The current stream is not currently capturing.");
  TORCH_CHECK(
      _currently_capturing_graphs.count(current_capture_id),
      "get_currently_capturing_graph() can be used only between capture_begin() and capture_end(). Did you use a stream without making it depend upon the original stream used for capture?");
  return _currently_capturing_graphs.at(current_capture_id);
}

void CUDAGraph::begin_capture_to_if_node(
    const at::Tensor& scalar_cuda_pred_tensor) {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  TORCH_CHECK(
      !has_graph_exec_,
      "begin_capture_to_if_node() must be called before capture_begin()");

  cudaStreamCaptureStatus status;
  cudaGraph_t currently_capturing_graph;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
      getCurrentCUDAStream(), &status, nullptr, &currently_capturing_graph));
  TORCH_CHECK(
      status == cudaStreamCaptureStatusActive,
      "capture_begin() must be called before begin_capture_to_if_node()");
  cudaGraphConditionalHandle handle;
  AT_CUDA_CHECK(cudaGraphConditionalHandleCreate(
      &handle, currently_capturing_graph, 0, 0));

  set_conditional_handle(handle, scalar_cuda_pred_tensor);

  const cudaGraphNode_t* dependencies;
  size_t num_dependencies;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
      getCurrentCUDAStream(),
      &status,
      nullptr,
      &currently_capturing_graph,
      &dependencies,
      &num_dependencies));
  TORCH_CHECK(status == cudaStreamCaptureStatusActive);

  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeIf;
  params.conditional.size = 1;

  cudaGraphNode_t cond_node;
  AT_CUDA_CHECK(cudaGraphAddNode(
      &cond_node,
      currently_capturing_graph,
      dependencies,
      num_dependencies,
      &params));

  cudaGraph_t if_node_child_graph = params.conditional.phGraph_out[0];

  AT_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      getCurrentCUDAStream(), &cond_node, 1, cudaStreamSetCaptureDependencies));

  UniquePtrExternalCudaStream child_stream = create_external_stream();
  conditional_graph_capture_streams_ids_.push(-1);
  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      capture_dev_, mempool_id_, create_child_allocate_filter());
  AT_CUDA_CHECK(cudaStreamBeginCaptureToGraph(
      *child_stream, if_node_child_graph, nullptr, nullptr, 0, capture_mode_));

  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
      *child_stream, &status, &conditional_graph_capture_streams_ids_.top()));
  TORCH_INTERNAL_ASSERT(
      status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  // We need to get the raw_stream here before emplace() to prevent
  // std::move(child_stream) from potentially executing before
  // *child_stream.
  cudaStream_t raw_stream = *child_stream;
  conditional_node_streams_.emplace(
      getStreamFromExternal(raw_stream, getCurrentCUDAStream().device_index()),
      std::move(child_stream));

  {
    std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
    _currently_capturing_graphs.emplace(
        conditional_graph_capture_streams_ids_.top(), this);
  }

#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
  return;
#endif
}

cudaGraphConditionalHandle CUDAGraph::begin_capture_to_while_loop_node(
    const at::Tensor& scalar_cuda_pred_tensor) {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  cudaStreamCaptureStatus status;
  cudaGraph_t currently_capturing_graph;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
      getCurrentCUDAStream(), &status, nullptr, &currently_capturing_graph));
  TORCH_CHECK(
      status == cudaStreamCaptureStatusActive,
      "capture_begin() must be called before begin_capture_to_while_loop_node()");
  cudaGraphConditionalHandle handle;
  AT_CUDA_CHECK(cudaGraphConditionalHandleCreate(
      &handle, currently_capturing_graph, 0, 0));

  set_conditional_handle(handle, scalar_cuda_pred_tensor);

  const cudaGraphNode_t* dependencies;
  size_t num_dependencies;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
      getCurrentCUDAStream(),
      &status,
      nullptr,
      &currently_capturing_graph,
      &dependencies,
      &num_dependencies));
  TORCH_CHECK(status == cudaStreamCaptureStatusActive);

  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeWhile;
  params.conditional.size = 1;

  cudaGraphNode_t cond_node;
  AT_CUDA_CHECK(cudaGraphAddNode(
      &cond_node,
      currently_capturing_graph,
      dependencies,
      num_dependencies,
      &params));

  cudaGraph_t while_node_child_graph = params.conditional.phGraph_out[0];

  AT_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      getCurrentCUDAStream(), &cond_node, 1, cudaStreamSetCaptureDependencies));

  UniquePtrExternalCudaStream child_stream = create_external_stream();
  conditional_graph_capture_streams_ids_.push(-1);
  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      capture_dev_, mempool_id_, create_child_allocate_filter());
  AT_CUDA_CHECK(cudaStreamBeginCaptureToGraph(
      *child_stream,
      while_node_child_graph,
      nullptr,
      nullptr,
      0,
      capture_mode_));

  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
      *child_stream, &status, &conditional_graph_capture_streams_ids_.top()));
  TORCH_INTERNAL_ASSERT(
      status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  // We need to get the raw_stream here before emplace() to prevent
  // std::move(child_stream) from potentially executing before
  // *child_stream.
  cudaStream_t raw_stream = *child_stream;
  conditional_node_streams_.emplace(
      getStreamFromExternal(raw_stream, getCurrentCUDAStream().device_index()),
      std::move(child_stream));

  {
    std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
    _currently_capturing_graphs.emplace(
        conditional_graph_capture_streams_ids_.top(), this);
  }

  return handle;
#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
  return cudaGraphConditionalHandle{};
#endif
}

void CUDAGraph::end_capture_to_conditional_node() {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  {
    std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
    CaptureId_t capture_id = conditional_graph_capture_streams_ids_.top();
    TORCH_CHECK(
        _currently_capturing_graphs.count(capture_id),
        "capture_end() called before capture_begin().");
    _currently_capturing_graphs.erase(capture_id);
  }

  CUDAStream stream = conditional_node_streams_.top().first.current_stream();
  cudaGraph_t graph;
  AT_CUDA_CHECK(cudaStreamEndCapture(stream.stream(), &graph));
  descendent_graphs_.push_back(graph);
  conditional_node_streams_.pop();
  conditional_graph_capture_streams_ids_.pop();

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
  if (conditional_graph_capture_streams_ids_.empty()) {
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        capture_dev_, mempool_id_, create_allocate_filter());
  } else {
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        capture_dev_, mempool_id_, create_child_allocate_filter());
  }
#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
#endif
}

std::function<bool(cudaStream_t)> CUDAGraph::create_allocate_filter() {
  return [this](cudaStream_t stream) {
    cudaStreamCaptureStatus status{};
    CaptureId_t stream_capture_id = 0;
    AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
    return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive && stream_capture_id == capture_id_;
  };
}

std::function<bool(cudaStream_t)> CUDAGraph::create_child_allocate_filter() {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  return [this, &current_capture_id = conditional_graph_capture_streams_ids_.top()](cudaStream_t stream) {
      cudaStreamCaptureStatus status;
      CaptureId_t stream_capture_id;
      AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
      return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive && stream_capture_id == current_capture_id;
  };
#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
  return std::function<bool(cudaStream_t)>();
#endif
}


} // namespace at::cuda

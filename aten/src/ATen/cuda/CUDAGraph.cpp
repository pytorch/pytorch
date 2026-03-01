#include <ATen/core/CachingHostAllocator.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/MemPool.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAFunctions.h>

#include <cstddef>

namespace at::cuda {

static bool _cuda_graphs_debug = false;

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
    mempool_id_ = at::cuda::MemPool::graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
  // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
  // due to the capture status being updated _after_ a capture had already started.
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, create_allocate_filter());

  auto filter = create_allocate_filter();

  at::getHostAllocator(at::kCUDA)->begin_allocate_to_pool(mempool_id_, [filter](c10::Stream stream) {
    return filter(CUDAStream(CUDAStream::UNCHECKED, stream));
  });

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

  TORCH_CHECK(stream.stream() == capture_stream_.stream(),
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
  at::getHostAllocator(at::kCUDA)->end_allocate_to_pool(mempool_id_);

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");

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

  capture_ended_ = true;
  has_graph_ = true;
  if (!keep_graph_) {
    instantiate();
    if (!_cuda_graphs_debug) {
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    }
    has_graph_ = false;
  }
}

void CUDAGraph::instantiate() {
  TORCH_CHECK(capture_ended_, "capture_end() must have been called before calling instantiate");

  if (has_graph_exec_) {
    TORCH_CHECK(keep_graph_, "instantiate() is intended to be called by the user only when keep_graph=true");
    AT_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
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
  has_graph_exec_ = true;
}

void CUDAGraph::replay() {
  TORCH_CHECK(capture_ended_,
              "Called CUDAGraph::replay without a preceding successful capture.");

  if (!has_graph_exec_) {
    TORCH_INTERNAL_ASSERT(keep_graph_);
    instantiate();
  }

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }
  // graph_exec_ may be replayed in any stream.
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));
}

void CUDAGraph::enable_debug_mode() {
  _cuda_graphs_debug = true;
}

void CUDAGraph::debug_dump(const std::string& debug_path) {
  if (_cuda_graphs_debug || keep_graph_) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling cudaGraphDebugDotPrint() with ", debug_path);
      C10_CUDA_CHECK_WARN(cudaGraphDebugDotPrint(graph_, debug_path.c_str(), cudaGraphDebugDotFlagsVerbose)); // most verbose output
      if (!keep_graph_) {
        AT_CUDA_CHECK(cudaGraphDestroy(graph_));
        has_graph_ = false;
      }
    }
  } else {
    TORCH_WARN("CUDA Graphs debug not enabled, set with [graph].enable_debug_mode()");
  }
}

cudaGraph_t CUDAGraph::raw_cuda_graph() {
  TORCH_CHECK(keep_graph_, "You cannot access the raw cudaGraph_t instance unless CUDAGraph was initialized with keep_graph=true");
  TORCH_CHECK(has_graph_, "You cannot access the raw cudaGraph_t instance until capture_end() has been called");
  return graph_;
}

cudaGraphExec_t CUDAGraph::raw_cuda_graph_exec() {
  TORCH_CHECK(
      has_graph_exec_,
      "You cannot access the raw cudaGraphExec_t instance until instantiate() has been called");
  return graph_exec_;
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
  if (capture_ended_) {
    // Clean up cuBLAS workspaces allocated on the capture stream, otherwise live allocations prevent
    // private pool cleanup
    clearCublasWorkspacesForStream(capture_stream_.stream());

    // notifyCaptureDestroy may throw. How should we handle this?
    c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
    at::getHostAllocator(at::kCUDA)->release_pool(mempool_id_);
    capture_ended_ = false;
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
  TORCH_CHECK(capture_ended_,
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
  cudaStreamCaptureStatus status{};
  CaptureId_t current_capture_id = 0;
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
      "This CUDAGraph instance already owns a captured graph.");

  TORCH_CHECK(!c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::graph_capture_record_stream_reuse(), "'graph_capture_record_stream_reuse:True' allocator config does not work with conditional control flow in a cuda graph today. See issue #175001 for updates");

  cudaStreamCaptureStatus status{};
  cudaGraph_t currently_capturing_graph{};
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
      getCurrentCUDAStream(), &status, nullptr, &currently_capturing_graph));
  TORCH_CHECK(
      status == cudaStreamCaptureStatusActive,
      "capture_begin() must be called before begin_capture_to_if_node()");
  cudaGraphConditionalHandle handle{};
  AT_CUDA_CHECK(cudaGraphConditionalHandleCreate(
      &handle, currently_capturing_graph, 0, 0));

  set_conditional_handle(handle, scalar_cuda_pred_tensor);

  const cudaGraphNode_t* dependencies{};
  const cudaGraphEdgeData* dependency_edges{};
  size_t num_dependencies = 0;
#if CUDA_VERSION >= 13000
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
      getCurrentCUDAStream(),
      &status,
      nullptr,
      &currently_capturing_graph,
      &dependencies,
      &dependency_edges,
      &num_dependencies));
#else
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo_v3(
      getCurrentCUDAStream(),
      &status,
      nullptr,
      &currently_capturing_graph,
      &dependencies,
      &dependency_edges,
      &num_dependencies
  ));
#endif
  TORCH_CHECK(status == cudaStreamCaptureStatusActive);

  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeIf;
  params.conditional.size = 1;

  cudaGraphNode_t cond_node{};
#if CUDA_VERSION >= 13000
  AT_CUDA_CHECK(cudaGraphAddNode(
      &cond_node,
      currently_capturing_graph,
      dependencies,
      dependency_edges,
      num_dependencies,
      &params));
#else
  AT_CUDA_CHECK(cudaGraphAddNode_v2(
      &cond_node,
      currently_capturing_graph,
      dependencies,
      dependency_edges,
      num_dependencies,
      &params));
#endif
  cudaGraph_t if_node_child_graph = params.conditional.phGraph_out[0];

#if CUDA_VERSION >= 13000
  AT_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
getCurrentCUDAStream(), &cond_node, nullptr, 1, cudaStreamSetCaptureDependencies));
#else
  AT_CUDA_CHECK(cudaStreamUpdateCaptureDependencies_v2(
getCurrentCUDAStream(), &cond_node, nullptr, 1, cudaStreamSetCaptureDependencies));
#endif

  CUDAStream child_stream = getStreamFromPool();
  conditional_graph_capture_ids_.push(0);
  conditional_rng_snapshots_.emplace();
  auto& conditional_rng_snapshot = conditional_rng_snapshots_.top();
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    conditional_rng_snapshot.emplace(
        generator_state, generator_state->offset_intragraph_);
  }

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
  at::getHostAllocator(at::kCUDA)->end_allocate_to_pool(mempool_id_);
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      capture_dev_, mempool_id_, create_child_allocate_filter());
  auto filter = create_child_allocate_filter();
  at::getHostAllocator(at::kCUDA)->begin_allocate_to_pool(mempool_id_, [filter](c10::Stream stream) {
    return filter(CUDAStream(CUDAStream::UNCHECKED, stream));
  });

  AT_CUDA_CHECK(cudaStreamBeginCaptureToGraph(
      child_stream, if_node_child_graph, nullptr, nullptr, 0, capture_mode_));

  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
      child_stream, &status, &conditional_graph_capture_ids_.top()));
  TORCH_INTERNAL_ASSERT(
      status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  conditional_node_streams_.emplace(child_stream);

  {
    std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
    _currently_capturing_graphs.emplace(
        conditional_graph_capture_ids_.top(), this);
  }

#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
  return;
#endif
}

void CUDAGraph::end_capture_to_conditional_node() {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  TORCH_INTERNAL_ASSERT(
      !conditional_rng_snapshots_.empty(),
      "Missing RNG snapshot for conditional node capture.");

  bool rng_or_generators_changed = false;
  auto& conditional_rng_snapshot = conditional_rng_snapshots_.top();
  if (conditional_rng_snapshot.size() != captured_generator_states_.size()) {
    rng_or_generators_changed = true;
  } else {
    for (const auto& [generator_state, offset_intragraph_before_capture] :
         conditional_rng_snapshot) {
      const auto generator_it = captured_generator_states_.find(generator_state);
      if (generator_it == captured_generator_states_.end() ||
          generator_state->offset_intragraph_ !=
              offset_intragraph_before_capture) {
        rng_or_generators_changed = true;
        break;
      }
    }
  }

  {
    std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
    CaptureId_t capture_id = conditional_graph_capture_ids_.top();
    TORCH_CHECK(
        _currently_capturing_graphs.count(capture_id),
        "capture_end() called before capture_begin().");
    _currently_capturing_graphs.erase(capture_id);
  }

  CUDAStream stream = conditional_node_streams_.top().current_stream();
  AT_CUDA_CHECK(cudaStreamEndCapture(stream.stream(), nullptr));
  conditional_node_streams_.pop();
  conditional_graph_capture_ids_.pop();
  conditional_rng_snapshots_.pop();

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
  at::getHostAllocator(at::kCUDA)->end_allocate_to_pool(mempool_id_);
  if (conditional_graph_capture_ids_.empty()) {
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        capture_dev_, mempool_id_, create_allocate_filter());
    auto filter = create_allocate_filter();
    at::getHostAllocator(at::kCUDA)->begin_allocate_to_pool(mempool_id_, [filter](c10::Stream stream) {
      return filter(CUDAStream(CUDAStream::UNCHECKED, stream));
    });
  } else {
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        capture_dev_, mempool_id_, create_child_allocate_filter());
    auto filter = create_child_allocate_filter();
    at::getHostAllocator(at::kCUDA)->begin_allocate_to_pool(mempool_id_, [filter](c10::Stream stream) {
      return filter(CUDAStream(CUDAStream::UNCHECKED, stream));
    });
  }

  constexpr const char* rng_with_conditional_nodes_error =
      "RNG within data-dependent conditional nodes is not supported yet.";
  TORCH_CHECK(!rng_or_generators_changed, rng_with_conditional_nodes_error);

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
  return [&current_capture_id = conditional_graph_capture_ids_.top()](cudaStream_t stream) {
      cudaStreamCaptureStatus status{};
      CaptureId_t stream_capture_id{};
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

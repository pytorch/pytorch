#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

namespace at::cuda {

static bool _cuda_graphs_debug = false;

MempoolId_t graph_pool_handle() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  // uuid count starts at 1. 0 is reserved to mean "wasn't set by graph_pool_handle".
  static std::atomic<CaptureId_t> uid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  return {0, uid++};
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
  return {0, 0};
#endif
}


// Get the expected id of a capture sequence so that we can call beginAllocateStreamToPool
// before starting a graph capture
CaptureId_t capture_sequence_id() {
  // id starts at 1:
  // Ensures uuid count starts at 1. 0 is reserved to mean "not set by cudaStreamGetCaptureInfo".
  // (But how do we know GetCaptureInfo never sets id_ to 0? Because that's the current behavior,
  // and I asked cuda devs to keep it that way, and they agreed.)
  static std::atomic<CaptureId_t> uuid{1};
  return uuid++;
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

CUDAGraph::CUDAGraph()
  // CUDAStreams may not be default-constructed.
  : capture_stream_(at::cuda::getCurrentCUDAStream()) {
#if (defined(USE_ROCM) && ROCM_VERSION < 50300)
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3");
#endif
}

void CUDAGraph::capture_begin(MempoolId_t pool/*=0*/, cudaStreamCaptureMode capture_mode) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  // For now, a CUDAGraph instance only accommodates the default generator on the device that's
  // current when capture begins. If any op in the captured region uses a non-default generator,
  // or a generator on another device, the offending generator will throw an error.
  // These restrictions simplify CUDAGraph, but could be relaxed in the future:
  // in principle, the underlying Cuda calls do permit cross-device ops to be captured.
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());

  auto options = TensorOptions().device(at::kCUDA).dtype(at::kLong);
  seed_extragraph_ = at::empty({1}, options);
  offset_extragraph_ = at::empty({1}, options);

  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  gen->capture_prologue(seed_extragraph_.data_ptr<int64_t>(), offset_extragraph_.mutable_data_ptr<int64_t>());

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_gen_ = gen;
  capture_dev_ = c10::cuda::current_device();

  id_ = capture_sequence_id();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our mempool_id_.
    // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
    mempool_id_ = {id_, 0};
  }

  // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
  // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
  // due to the capture status being updated _after_ a capture had already started.
  c10::cuda::CUDACachingAllocator::beginAllocateStreamToPool(capture_dev_, capture_stream_, mempool_id_);

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));

  cudaStreamCaptureStatus status;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  TORCH_INTERNAL_ASSERT(id_ > 0);
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

void CUDAGraph::capture_end() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));

  c10::cuda::CUDACachingAllocator::endAllocateStreamToPool(capture_dev_, capture_stream_);

  TORCH_CHECK(graph_ != NULL, "Invalid capture.");
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
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  int version;
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
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  } else {
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch));
  }
#endif

  has_graph_exec_ = true;

  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  TORCH_CHECK(gen == capture_gen_,
              "Default CUDA RNG generator on current device at capture end "
              "is different from default generator on current device "
              "when capture began");
  wholegraph_increment_ = gen->capture_epilogue();

  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, NULL, &numCUDAGraphNodes));
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
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

void CUDAGraph::replay() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  // Just like any RNG consumer kernel!
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(wholegraph_increment_);
  }
  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  offset_extragraph_.fill_(int64_t(rng_engine_inputs.offset_.val));

  // graph_exec_ may be replayed in any stream.
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    // Workaround for bug in libcuda.so that causes replayed graphs with
    // certain topologies to be corrupted (kernels elided, internal syncs
    // ignored) when replayed back to back without a sync in between.
    // The bug is fixed in CUDA 11.4+.
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
#else
  TORCH_CHECK(false, "CUDA graphs is not yet supported on ROCM");
#endif
}

void CUDAGraph::enable_debug_mode() {
#if (!defined(USE_ROCM) || ROCM_VERSION >= 50300) || (defined(USE_ROCM) && ROCM_VERSION >= 50600)
  _cuda_graphs_debug = true;
#else
  TORCH_CHECK(false, "CUDA graphs is not yet supported on ROCM");
#endif

}

void CUDAGraph::debug_dump(const std::string& debug_path) {
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11030)|| (defined(USE_ROCM) && ROCM_VERSION >= 50600)
  if (_cuda_graphs_debug) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling cudaGraphDebugDotPrint() with ", debug_path);
      C10_CUDA_CHECK_WARN(cudaGraphDebugDotPrint(graph_, debug_path.c_str(), 1<<10)); // most verbose output
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    }
  } else {
    TORCH_WARN("CUDA Graphs debug not enabled, set with torch._C._cuda_enable_graphs_debug_mode");
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.3 or ROCM >= 5.6");
#endif
}

void CUDAGraph::reset() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
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
  }
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t CUDAGraph::pool() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
  return mempool_id_;
}

CUDAGraph::~CUDAGraph() {
  reset();
}

} // namespace at::cuda

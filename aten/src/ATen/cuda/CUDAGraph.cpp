#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

namespace at {
namespace cuda {

MempoolId_t graph_pool_handle() {
#if CUDA_VERSION >= 11000
  // uuid count starts at 1. 0 is reserved to mean "wasn't set by graph_pool_handle".
  static std::atomic<CaptureId_t> uuid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  return {0, uuid++};
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0");
  return {0, 0};
#endif
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
#if CUDA_VERSION < 11000
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0");
#endif
}

void CUDAGraph::capture_begin(MempoolId_t pool/*=0*/) {
#if CUDA_VERSION >= 11000
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
  offset_extragraph_ = at::empty({1}, options);

  gen->capture_prologue(offset_extragraph_.data_ptr<int64_t>());

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_gen_ = gen;
  capture_dev_ = c10::cuda::current_device();

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal));

  // Stashes the current capture's uuid.
  cudaStreamCaptureStatus status;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  // Ensures uuid count starts at 1. 0 is reserved to mean "not set by cudaStreamGetCaptureInfo".
  // (But how do we know GetCaptureInfo never sets id_ to 0? Because that's the current behavior,
  // and I asked cuda devs to keep it that way, and they agreed.)
  TORCH_INTERNAL_ASSERT(id_ > 0);
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

  // When CUDACachingAllocator allocates while a capture is underway, it calls cudaStreamGetCaptureInfo
  // to get the current stream's capture id, if any. Here we tell CUDACachingAllocator: if the stream
  // has a capture id matching this graph's id_, use the private pool mempool_id_ identifies.
  //
  // There's a small chance of a bad allocation here if another thread launches a kernel on
  // capture_stream_ between the call to cudaStreamBeginCapture above and the call to
  // notifyCaptureBegin below.
  // But I don't think we need to worry about it because that use case makes no sense:
  // The user has no business launching kernels on capture_stream_ from another thread
  // while calling capture_begin. They'll have no idea if their side thread's
  // kernel will end up as part of the capture or not.
  c10::cuda::CUDACachingAllocator::notifyCaptureBegin(capture_dev_, id_, mempool_id_);
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0");
#endif
}

void CUDAGraph::capture_end() {
#if CUDA_VERSION >= 11000
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  c10::cuda::CUDACachingAllocator::notifyCaptureEnd(capture_dev_, id_);

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));
  TORCH_CHECK(graph_ != NULL, "Invalid capture.");
  has_graph_ = true;

  // Trailing NULL, NULL, 0 arguments were recommended by Cuda driver people,
  // who prefer not to report error message through these arguments moving forward
  // (they prefer return value, or errors on api calls internal to the capture)
  AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
  has_graph_exec_ = true;

  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  TORCH_CHECK(gen == capture_gen_,
              "Default CUDA RNG generator on current device at capture end "
              "is different from default generator on current device "
              "when capture began");
  wholegraph_increment_ = gen->capture_epilogue();

  // Now that we've instantiated graph_ into graph_exec_,
  // we don't need graph_ anymore.
  AT_CUDA_CHECK(cudaGraphDestroy(graph_));
  has_graph_ = false;
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0");
#endif
}

void CUDAGraph::replay() {
#if CUDA_VERSION >= 11000
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture.");

  {
    c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

    // Just like any RNG consumer kernel!
    auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
        c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
    PhiloxCudaState rng_engine_inputs;
    {
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_cuda_state(wholegraph_increment_);
    }
    offset_extragraph_.fill_(int64_t(rng_engine_inputs.offset_.val));

    // graph_exec_ may be replayed in any stream.
    AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0");
#endif
}

void CUDAGraph::reset() {
#if CUDA_VERSION >= 11000
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
  // a Juptyer notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.
  if (has_graph_ || has_graph_exec_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10::cuda::CUDACachingAllocator::notifyCaptureDestroy(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
  }
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0");
#endif
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t CUDAGraph::pool() {
#if CUDA_VERSION >= 11000
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0");
#endif
  return mempool_id_;
}

CUDAGraph::~CUDAGraph() {
  reset();
}

} // namespace cuda
} // namespace at

#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAFunctions.h>

namespace at {
namespace cuda {

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

CUDAGraph::CUDAGraph()
  // CUDAStreams may not be default-constructed.
  : capture_stream_(at::cuda::getCurrentCUDAStream()) {
#if CUDA_VERSION < 11000
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0");
#endif
}

void CUDAGraph::capture_begin() {
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

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal));

  // Stashes the current graph's uuid.
  cudaStreamCaptureStatus status;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0");
#endif
}

void CUDAGraph::capture_end() {
#if CUDA_VERSION >= 11000
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

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

CUDAGraph::~CUDAGraph() {
  reset();
}

} // namespace cuda
} // namespace at

#include <ATen/cuda/Exceptions.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAFunctions.h>

namespace at {

// forward-declares empty
namespace native {
  Tensor empty(IntArrayRef size, const TensorOptions& options);
}

namespace cuda {

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
  const auto& gen = at::cuda::detail::getDefaultCUDAGenerator(c10::cuda::current_device());

  auto options = TensorOptions().device(at::kCUDA).dtype(at::kLong);
  offset_extragraph_ = at::native::empty({1}, options);

  gen.capture_prologue(offset_extragraph_.data_ptr<int64_t>());

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;

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

  const auto& gen = at::cuda::detail::getDefaultCUDAGenerator(c10::cuda::current_device());
  wholegraph_increment_ = gen.capture_epilogue();

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
    const auto& gen = at::cuda::detail::getDefaultCUDAGenerator(c10::cuda::current_device());
    // Just like any RNG consumer kernel!
    PhiloxCudaState rng_engine_inputs;
    {
      std::lock_guard<std::mutex> lock(gen.mutex_);
      rng_engine_inputs = gen.philox_cuda_state(wholegraph_increment_);
    }
    offset_extragraph_.fill_(rng_engine_inputs.offset_.val);

    // graph_exec_ may be replayed in any stream.
    AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0");
#endif
}

// This can throw, but my only alternative idea is to have a Python-side
// wrapper with a __del__ method on the Python side that calls e.g.
// void CUDAGraph::drop_graph() {
//   AT_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
// }
// Stackoverflow appears to hate __del__ as much as throwing in destructors.
CUDAGraph::~CUDAGraph() {
#if CUDA_VERSION >= 11000
  // these checks reduce (but can't eliminate) the chance of throwing an exception.
  if (has_graph_) {
    AT_CUDA_CHECK(cudaGraphDestroy(graph_));
  }
  if (has_graph_exec_) {
    AT_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
  }
#endif
}

} // namespace cuda
} // namespace at

#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/StreamGuard.h>

void CUDAGraph::capture_begin() {}
// void CUDAGraph::capture_begin() {
//   TORCH_CHECK(!has_capture_,
//               "This CUDAGraph instance already owns a captured graph. "
//               "To capture a new graph, create a new instance or call "
//               "drop_graph() on the present instance.");
// 
//   for (DeviceIndex idx = 0; idx < c10::cuda::device_count(); idx++) {
//     const auto& gen = at::cuda::detail::getDefaultCUDAGenerator(idx);
//     gen.capture_prologue(this);
//   }
// 
//   // cudaStreamCaptureModeGlobal is the most conservative option to
//   // prevent potentially unsafe CUDA API calls during capture.  See
//   // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
//   AT_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
// 
//   // Stashes the current graph's uuid.
//   cudaStreamCaptureStatus status;
//   AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &id_);
//   TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);
// }

void CUDAGraph::capture_end() {}
// CUDAGraph::capture_end() {
//   auto stream = at::cuda::getCurrentCUDAStream():
//   TORCH_CHECK(stream == capture_stream_,
//               "Capture must end on the same stream it was initiated.")
//   AT_CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
//   TORCH_CHECK(graph_ != NULL, "Invalid capture."); 
// 
//   // Trailing NULL, NULL, 0 arguments recommended by Cuda driver people.
//   // They prefer to deliver error messages differently moving forward
//   // (e.g. return value, or errors on api calls internal to the capture)
//   AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
// 
//   for (DeviceIndex idx = 0; idx < c10::cuda::device_count(); idx++) {
//     const auto& gen = at::cuda::detail::getDefaultCUDAGenerator(idx);
//     gen.capture_epilogue(this);
//   }
// 
//   // Now that we've instantiated graph_ into graph_exec_,
//   // we don't need graph_ anymore.
//   AT_CUDA_CHECK(cudaGraphDestroy(graph_));
//   has_capture_ = true;
// }

void CUDAGraph::replay() {}
// void CUDAGraph::replay() {
//   TORCH_CHECK(has_capture_,
//               "Called replay() on a CUDAGraph object before capturing.");
// 
//   for (const auto& used : used_rng_) {
//     const auto& dev = std::get<0>(used);
//     auto& offset_tensor = std::get<1>(used);
//     const auto& gen = at::cuda::detail::getDefaultCUDAGenerator(dev);
//     // Just like any RNG consumer kernel!
//     PhiloxCudaState rng_engine_inputs;
//     {
//       std::lock_guard<std::mutex> lock(gen->mutex_);
//       rng_engine_inputs = gen.philox_cuda_state();
//     }
//     offset_storage.fill_(state.offset_.val);
//   }
//  
//   // The graph may be replayed in any stream. 
//   AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));
// }

// I'd like to put this in ~CUDAGraph but it can throw exceptions.
void CUDAGraph::drop_graph() {}
// void CUDAGraph::drop_graph() {
//   AT_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
//   has_capture_ = false;
// }

at::Tensor CUDAGraph::generator_callback(DeviceIndex) {
  return {};
}

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>

// global typedef but i don't see any cops
typedef unsigned long long CUDAGraphid_t;

namespace at {

class CUDAGeneratorImpl;

namespace cuda {

struct TORCH_CUDA_CPP_API CUDAGraph {
  CUDAGraph();
  ~CUDAGraph();

  void capture_begin(CUDAGraphid_t pool=0);
  void capture_end();
  void replay();
  void reset();
  CUDAGraphid_t pool();

  protected:
#if CUDA_VERSION >= 11000
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
#endif

  // internal states for error checking
  bool has_graph_ = false;
  bool has_graph_exec_ = false;

  // uuid, retrieved from Cuda
  CUDAGraphid_t id_;

  // uuid used to request a particular private mempool from CUDACachingAllocator.
  // By default, this will be set to id_, but if capture_begin is called with
  // "pool=other_graph.pool()", this graph's mempool_id_ will be set to the other
  // graph's mempool_id_, and therefore share a mempool with that other graph.
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  CUDAGraphid_t mempool_id_;

  // Stream on which capture began
  at::cuda::CUDAStream capture_stream_;

  // Default generator on device where capture began
  at::CUDAGeneratorImpl* capture_gen_;

  // RNG state trackers
  at::Tensor offset_extragraph_;
  CUDAGraphid_t wholegraph_increment_;
};

// RAII guard for "cudaStreamCaptureMode", a thread-local value
// that controls the error-checking strictness of a capture.
struct TORCH_CUDA_CPP_API cudaStreamCaptureModeGuard {
  cudaStreamCaptureModeGuard(cudaStreamCaptureMode desired) {
#if CUDA_VERSION >= 11000
    strictness_ = desired;
    cudaThreadExchangeStreamCaptureMode(&strictness_);
#endif
  }
  ~cudaStreamCaptureModeGuard() {
#if CUDA_VERSION >= 11000
    cudaThreadExchangeStreamCaptureMode(&strictness_);
#endif
  }

  private:
  cudaStreamCaptureMode strictness_;
};

} // namespace cuda
} // namespace at

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>

namespace at {

class CUDAGeneratorImpl;

namespace cuda {

struct TORCH_CUDA_CPP_API CUDAGraph {
  CUDAGraph();
  ~CUDAGraph();

  void capture_begin(CaptureId_t pool=0);
  void capture_end();
  void replay();
  void reset();
  CaptureId_t pool();

  protected:
#if CUDA_VERSION >= 11000
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
#endif

  // internal states for error checking
  bool has_graph_ = false;
  bool has_graph_exec_ = false;

  // uuid of this instance's current capture, retrieved from Cuda
  CaptureId_t id_;

  // uuid used to request a particular private mempool from CUDACachingAllocator.
  // By default, this will be set to id_, but if capture_begin is called with
  // "pool=other_graph.pool()", this graph's mempool_id_ will be set to the other
  // graph's mempool_id_, and therefore share a mempool with that other graph.
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  CaptureId_t mempool_id_;

  // Stream on which capture began
  at::cuda::CUDAStream capture_stream_;

  // Default generator on device where capture began
  at::CUDAGeneratorImpl* capture_gen_;

  // Device where capture occurred. Right now, for simplicity, we require all ops
  // in a capture to run on the same device, but this is a limitation of CUDAGraph,
  // not CUDA itself.  We can straightforwardly modify CUDAGraph to support multi-device
  // captures if needed.
  int capture_dev_;

  // RNG state trackers
  at::Tensor offset_extragraph_;
  uint64_t wholegraph_increment_;
};

} // namespace cuda
} // namespace at

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>

#include <mutex>

namespace at {

struct CUDAGeneratorImpl;

namespace cuda {

// Standalone way to get a unique mempool id usable as a pool=... argument
// to CUDAGraph::capture_begin
TORCH_CUDA_CPP_API MempoolId_t graph_pool_handle();

struct TORCH_CUDA_CPP_API CUDAGraph {
  CUDAGraph();
  ~CUDAGraph();

  static void inc_pending_event_queries();
  static void dec_pending_event_queries();
  static int num_pending_event_queries();
  void capture_begin(MempoolId_t pool={0, 0}, cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal);
  void capture_end();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);

  protected:
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
#endif

  static std::atomic<int> pending_event_queries;

  // internal states so reset() can do its best cleaning up
  // Set to true in capture_end if cudaStreamEndCapture succeeded
  // Set back to false soon after, when graph_ is consumed by cudaGraphInstantiate
  // to create graph_exec_, then graph_ is deleted
  bool has_graph_ = false;
  // Set to true in capture_end if cudaGraphInstantiate succeeded
  bool has_graph_exec_ = false;

  // uuid of this instance's current capture, used to
  // specify the pool.
  CaptureId_t id_;

  // the ID assigned by cuda during graph capture,
  // used to identify when a stream is participating in capture
  CaptureId_t capture_id_ = -1;

  // uuid used to request a particular private mempool from CUDACachingAllocator.
  // By default, this will be set to {id_, 0}.
  //
  // If capture_begin is called with "pool=other_graph.pool()", this graph's mempool_id_
  // will be set to the other graph's mempool_id_, and therefore share a mempool with the
  // other graph.
  //
  // If capture_begin is called with "pool=handle" where "handle" came from graph_pool_handle(),
  // it will share a mempool with any other captures that used "pool=handle".
  //
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  MempoolId_t mempool_id_;

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
  at::Tensor seed_extragraph_;
  at::Tensor offset_extragraph_;
  uint64_t wholegraph_increment_;
};

} // namespace cuda
} // namespace at

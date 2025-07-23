#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/flat_hash_map.h>

namespace at {

struct Generator;
struct CUDAGeneratorImpl;
struct CUDAGeneratorState;

namespace cuda {

// Standalone way to get a unique mempool id usable as a pool=... argument
// to CUDAGraph::capture_begin
TORCH_CUDA_CPP_API MempoolId_t graph_pool_handle();

struct TORCH_CUDA_CPP_API CUDAGraph {
  CUDAGraph(bool keep_graph=false);
  ~CUDAGraph();

  // See Note [Explicit Registration of Generators to the CUDA Graph]
  void register_generator_state(c10::intrusive_ptr<at::CUDAGeneratorState> state);
  void register_generator_state(const at::Generator& generator);
  void capture_begin(
      MempoolId_t pool = {0, 0},
      cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal);
  void capture_end();
  void instantiate();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);
  cudaGraph_t raw_cuda_graph();

 protected:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;

  // internal states so reset() can do its best cleaning up

  // Set to true in capture_end if cudaStreamEndCapture succeeded
  // Set back to false after instantiate() unless keep_graph=True or
  // enable_debug_mode() was called on any CUDAGraph instance.
  bool has_graph_ = false;
  // Set to true in capture_end if cudaStreamEndCapture succeeded
  bool capture_ended_ = false;
  // Set to true in capture_end if cudaGraphInstantiate succeeded
  bool has_graph_exec_ = false;

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

  // multiple generator states and their wholegraph_increments in this graph
  // that are managed by the CUDA Graph
  ska::flat_hash_map<c10::intrusive_ptr<at::CUDAGeneratorState>, uint64_t>
      captured_generator_states_;

  // Device where capture occurred. Right now, for simplicity, we require all ops
  // in a capture to run on the same device, but this is a limitation of CUDAGraph,
  // not CUDA itself.  We can straightforwardly modify CUDAGraph to support multi-device
  // captures if needed.
  // init capture_dev_ as UNDEFINED_DEVICE to check that it stores the real device id in the destructor
  static constexpr c10::DeviceIndex UNDEFINED_DEVICE = -1;
  c10::DeviceIndex capture_dev_{UNDEFINED_DEVICE};

  bool keep_graph_;
};

} // namespace cuda
} // namespace at

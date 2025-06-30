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


template<bool VOLTA_OR_LATER>
struct KernelUpdateSOA {
  // Even on Cuda 12.4, Torch supports compute capability down to sm_50
  // However, large parameter support was only added in sm_70
  // https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/
  static constexpr size_t KERNEL_PARAM_LIMIT_BYTES = VOLTA_OR_LATER ? 32764 : 4096;
  static constexpr size_t PER_UPDATE = sizeof(void*) + sizeof(size_t) + sizeof(cudaGraphDeviceNode_t);

  static constexpr size_t MAX_NUM_UPDATES = (KERNEL_PARAM_LIMIT_BYTES - sizeof(size_t)) / PER_UPDATE;

  size_t num_updates;
  // struct-of-arrays pattern for global memory coalescing
  cudaGraphDeviceNode_t device_nodes[MAX_NUM_UPDATES];
  void* new_pointers[MAX_NUM_UPDATES];
  size_t param_offsets[MAX_NUM_UPDATES];
};

template<bool VOLTA_OR_LATER>
void dynamic_graph_updater(const KernelUpdateSOA<VOLTA_OR_LATER>& updates);

struct DynamicGraphKernelParamUpdate {
  // in other words:
  // dev_node.params[param_offset] = allocations[alloc_idx] + offset

  cudaGraphDeviceNode_t dev_node;
  size_t param_offset; // which arg are we updating
  size_t alloc_idx; // which allocation is it
  size_t offset; // how deep into the allocation?
};

struct DynamicGraphOtherNodeUpdate {
  // for the (rare) case where a cudaMemsetAsync or cudaMemcpyAsync involved a dynamic tensor
  cudaGraphNode_t node;
  std::function<cudaGraphNodeParams(std::vector<void*>)> compute_new_params;
};

struct HostMemoryUpdate {
  size_t alloc_idx;
  void **address_to_update;
  size_t offset;
};

struct DynamicGraphAllocation {
  char* ptr; // device-side pointer to the allocation
  size_t size; // size of the allocation
  size_t alloc_idx; // which allocation was this originally
};

struct TORCH_CUDA_CPP_API CUDAGraph {
  CUDAGraph(bool keep_graph=false);
  ~CUDAGraph();

  // See Note [Explicit Registration of Generators to the CUDA Graph]
  void register_generator_state(c10::intrusive_ptr<at::CUDAGeneratorState> state);
  void register_generator_state(const at::Generator& generator);
  void capture_begin(
      MempoolId_t pool = {0, 0},
      cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal,
      bool dynamic_graph = false);
  void capture_end();
  void instantiate();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);
  cudaGraph_t raw_cuda_graph();
  void become_dynamic(const std::vector<at::Tensor>& dynamic_tensors);
  void replay_dynamic(const std::vector<at::Tensor>& dynamic_tensors);

  TORCH_CUDA_CPP_API friend bool operator==(const CUDAGraph& left, const CUDAGraph& right);
  TORCH_CUDA_CPP_API friend bool operator!=(const CUDAGraph& left, const CUDAGraph& right);
  static std::shared_ptr<c10::Allocator> get_mem_allocator();

 protected:
  void add_dynamic_update(const std::tuple<size_t, size_t, size_t>& result, cudaGraphNode_t node, size_t param_offset);
  void add_host_memory_update(size_t alloc_idx, void **address_to_update, size_t offset);
  template<bool VOLTA_OR_LATER>
  void launch_dynamic_updaters(const std::vector<void*>& actualDataPtrs);
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

  // Will this graph have dynamic input/output tensors?
  bool dynamic_graph_;
  // (if dynamic) Which allocations are dynamic?
  std::vector<DynamicGraphAllocation> allocations_;
  // (if dynamic) Which parameters of which kernels need to be updated to the input tensors or new allocations?
  std::vector<DynamicGraphKernelParamUpdate> kernel_param_updates_;
  // (if dynamic) Some Torch ops use cudaMemcpyAsync or cudaMemsetAsync, those also need to have their pointers updated
  std::vector<DynamicGraphOtherNodeUpdate> graph_node_param_updates_;
  std::vector<HostMemoryUpdate> host_memory_updates_;

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

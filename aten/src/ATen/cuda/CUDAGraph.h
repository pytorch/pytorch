#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/flat_hash_map.h>

#include <limits>
#include <optional>
#include <stack>
#include <vector>

#if defined(USE_ROCM) || !(defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
// this type is not defined until CUDA 12.4, but we use it as a
// parameter type and return type in some below functions, so we give
// it the same definition as in CUDA 12.4.
typedef unsigned long long cudaGraphConditionalHandle;
#endif // defined(USE_ROCM) || !(defined(CUDA_VERSION) && CUDA_VERSION >= 12040)

namespace at {

struct Generator;
struct CUDAGeneratorImpl;
struct CUDAGeneratorState;

namespace cuda {

// Standalone way to get a unique mempool id usable as a pool=... argument
// to CUDAGraph::capture_begin
TORCH_CUDA_CPP_API MempoolId_t graph_pool_handle();

// Returns true if any CUDAGraph capture is currently active in this process.
// Used by ProcessGroupNCCL's ROCm watchdog workaround to avoid calling
// hipEventQuery during active capture on HIP runtimes without the
// event-query capture-mode fix (https://github.com/ROCm/clr/pull/3176).
// Not needed on CUDA/NVIDIA where cross-thread event query does not have this
// restriction.
#if defined(USE_ROCM)
TORCH_CUDA_CPP_API bool is_graph_capture_active();
#endif // defined(USE_ROCM)

struct CUDAGraph;

TORCH_CUDA_CPP_API CUDAGraph* get_graph_from_capture_id(CaptureId_t capture_id);

struct TORCH_CUDA_CPP_API CUDAGraph {
  CUDAGraph(bool keep_graph=false);
  ~CUDAGraph();

  // Copy and move constructors and assignments are disabled. These
  // were disabled because pybind11 believed that CUDAGraph was copy
  // constructable because
  // pybind11::is_copy_constructible<CUDAGraph>::value originally
  // evaluated to true. However, it cannot generate a copy constructor
  // because CUDAGeneratorState, one of CUDAGraph's members, is an
  // incomplete type unless CUDAGeneratorImpl.h is included. However,
  // that would create a circular dependency between
  // CUDAGeneratorImpl.h and CUDAGraph.h. Disabling the copy and move
  // constructors is the most straightforward way to prevent pybind11
  // from trying to generate default implementations of them.
  //
  // We needed pybind11 to return a reference to a CUDAGraph as part
  // of wrapping CUDAGraph::get_currently_capturing_graph, which
  // unearthed the above problem.
  CUDAGraph(const CUDAGraph&) = delete;
  CUDAGraph& operator=(const CUDAGraph&) = delete;
  CUDAGraph(CUDAGraph&& other) = delete;
  CUDAGraph& operator=(CUDAGraph&& other) = delete;

  void register_generator_state(c10::intrusive_ptr<at::CUDAGeneratorState> state);
  void capture_begin(
      MempoolId_t pool = {0, 0},
      cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal);
  void capture_end();
  // Split capture_end: capture_end_pre ends capture leaving graph_ live (both
  // keep_graph modes); capture_end_post finalizes (instantiate + destroy for
  // keep_graph=false). capture_end() == pre() + post(). The split lets callers
  // operate on the captured cudaGraph_t before finalization.
  void capture_end_pre();
  void capture_end_post();
  void instantiate();
  // True once the cudaGraphExec_t has been instantiated (by capture_end when
  // keep_graph=false, or by an explicit instantiate()). The Python replay()
  // wrapper uses this to instantiate on demand for keep_graph=true.
  bool has_graph_exec() const {
    return has_graph_exec_;
  }
  void replay();
  void reset();
  MempoolId_t pool();
  std::vector<MempoolId_t> pools();
  void retain_pool(MempoolId_t pool);
  void enable_debug_mode();
  cudaGraph_t raw_cuda_graph();
  cudaGraphExec_t raw_cuda_graph_exec();

  static CUDAGraph* get_currently_capturing_graph();
  void begin_capture_to_if_node(const Tensor& scalar_cuda_pred_tensor);
  void begin_capture_to_while_node(const Tensor& scalar_cuda_pred_tensor);
  void end_capture_to_conditional_node();
  void set_conditional_handle_for_current_node(
      const Tensor& scalar_cuda_pred_tensor);
  static void set_conditional_handle(
      cudaGraphConditionalHandle handle,
      const Tensor& scalar_cuda_pred_tensor);

 private:
  template <typename StreamType>
  std::function<bool(StreamType)> create_allocate_filter() const;
  std::function<bool(cudaStream_t)> create_child_allocate_filter();
  void record_retained_pool(MempoolId_t pool);
  bool has_retained_pool(MempoolId_t pool) const;
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  void begin_capture_to_conditional_node(
      const Tensor& scalar_cuda_pred_tensor,
      cudaGraphConditionalNodeType conditional_type);
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12040

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

  // Set to true in capture_begin once a private pool has been acquired
  // (beginAllocateToPool). Tells reset() it must release the pool, even if the
  // capture failed before capture_end() completed. Otherwise a failed capture
  // leaks the pool: its use_count never returns to zero, so empty_cache can
  // never reclaim its segments for the rest of the process.
  bool allocated_pool_ = false;
  // Set to true in capture_begin after beginAllocateToPool and cleared in
  // capture_end after endAllocateToPool. Tells reset() whether the allocator is
  // still routing allocations to the pool (capture abandoned before capture_end
  // ran) and must be ended before the pool can be released.
  bool capturing_to_pool_ = false;

  // the ID assigned by cuda during graph capture,
  // used to identify when a stream is participating in capture
  CaptureId_t capture_id_ = 0;

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
  std::vector<MempoolId_t> retained_mempool_ids_;

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
  cudaStreamCaptureMode capture_mode_{};

#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  struct OwnedCUDAStream {
    cudaStream_t stream = nullptr;
    OwnedCUDAStream() = default;
    explicit OwnedCUDAStream(cudaStream_t s) : stream(s) {}
    ~OwnedCUDAStream() {
      if (stream)
        C10_CUDA_CHECK_WARN(cudaStreamDestroy(stream));
    }
    OwnedCUDAStream(const OwnedCUDAStream&) = delete;
    OwnedCUDAStream& operator=(const OwnedCUDAStream&) = delete;
    OwnedCUDAStream(OwnedCUDAStream&& o) noexcept
        : stream(std::exchange(o.stream, nullptr)) {}
    OwnedCUDAStream& operator=(OwnedCUDAStream&& o) noexcept {
      if (stream)
        C10_CUDA_CHECK_WARN(cudaStreamDestroy(stream));
      stream = std::exchange(o.stream, nullptr);
      return *this;
    }
  };

  std::stack<at::cuda::CUDAStreamGuard> conditional_node_streams_;
  std::stack<CaptureId_t> conditional_graph_capture_ids_;
  std::stack<OwnedCUDAStream> conditional_node_raw_streams_;
  std::stack<cudaGraphConditionalHandle> conditional_node_handles_;
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12040
};

template <>
std::function<bool(cudaStream_t)> CUDAGraph::create_allocate_filter<cudaStream_t>() const;
template <>
std::function<bool(c10::Stream)> CUDAGraph::create_allocate_filter<c10::Stream>() const;

} // namespace cuda
} // namespace at

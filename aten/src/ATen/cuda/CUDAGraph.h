#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/CUDAGeneratorImpl.h>

namespace at {
namespace cuda {

struct TORCH_CUDA_API CUDAGraph {
  CUDAGraph();
  ~CUDAGraph();

  void capture_begin();
  void capture_end();
  void replay();
  void reset();

  protected:
#if CUDA_VERSION >= 11000
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
#endif

  // internal states for error checking
  bool has_graph_ = false;
  bool has_graph_exec_ = false;

  // uuid, retrieved from Cuda
  unsigned long long id_;

  // Stream on which capture began
  at::cuda::CUDAStream capture_stream_;

  // Default generator on device where capture began
  at::CUDAGeneratorImpl* capture_gen_;

  // RNG state trackers
  at::Tensor offset_extragraph_;
  uint64_t wholegraph_increment_;
};

} // namespace cuda
} // namespace at

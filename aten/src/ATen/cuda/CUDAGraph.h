#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>

namespace at {
namespace cuda {

struct TORCH_CUDA_API CUDAGraph {
  CUDAGraph();
  ~CUDAGraph();

  void capture_begin();
  void capture_end();
  void replay();

  protected:
  #if CUDA_VERSION >= 11000
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
  #endif
  bool has_graph_ = false;
  bool has_graph_exec_ = false;
  unsigned long long id_;
  at::cuda::CUDAStream capture_stream_;
  at::Tensor offset_extragraph_;
  uint64_t wholegraph_increment_;
};

} // namespace cuda
} // namespace at

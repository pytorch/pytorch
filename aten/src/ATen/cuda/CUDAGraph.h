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
  at::Tensor generator_callback(DeviceIndex);

  protected:
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
  bool has_graph_ = false;
  bool has_graph_exec_ = false;
  std::vector<std::tuple<DeviceIndex, at::Tensor>> used_rng_;
  unsigned long long id_;
  at::cuda::CUDAStream capture_stream_;
};

} // namespace cuda
} // namespace at

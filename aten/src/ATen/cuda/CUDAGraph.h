#include <ATen/Tensor.h>

class CUDAGraph {
  CUDAGraph() = default;
  ~CUDAGraph() = default;

  void capture_begin();
  void capture_end();
  void replay()
  void drop_graph();
  at::Tensor generator_callback(DeviceIndex);

  private:
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
  bool has_capture_ = false;
  std::vector<std::tuple<DeviceIndex, at::Tensor>> used_rng_;
  uint64_t id_;
}

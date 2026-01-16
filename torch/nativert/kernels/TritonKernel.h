#pragma once

#include <c10/core/Device.h>

#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/executor/triton/TritonKernelManager.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

class TritonKernel : public OpKernel {
 public:
  TritonKernel() = delete;
  TritonKernel(
      const Node* node,
      caffe2::serialize::PyTorchStreamReader* reader);
  ~TritonKernel() override;

  void computeInternal(ExecutionFrame& executionFrame) const override;

 private:
  std::unique_ptr<TritonKernelManager> loader_;

  // unnamed node attributes will be passed as arguments to the kernel
  std::vector<void*> attr_ptrs_;
  // Storage for float attributes that were serialized as doubles
  std::vector<float> float_attrs_;
  std::vector<int64_t> output_indices_;
  LaunchParams launch_params_;
};

} // namespace torch::nativert

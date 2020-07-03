#pragma once

#include <memory>

#include <ATen/ATen.h>
#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/utils/pybind.h>

namespace c10d {

// Broadcast many tensors to all processes in the process group.
void broadcast_coalesced(
    std::shared_ptr<c10d::ProcessGroup> process_group,
    at::TensorList tensors,
    size_t buffer_size);

class GradBucket {
 public:
  explicit GradBucket(std::vector<at::Tensor>& tensors);
  std::vector<at::Tensor> tensors_;
};

struct CommHookInterface {
 public:
  virtual c10::intrusive_ptr<torch::jit::Future> operate(
      const GradBucket& bucket) = 0;
};

class TORCH_API PythonCommHook : public CommHookInterface {
 public:
  PythonCommHook(py::object state, py::object hook);

  c10::intrusive_ptr<torch::jit::Future> operate(
      const GradBucket& bucket) override;

 private:
  py::object state_;
  py::object hook_;
};
} // namespace c10d

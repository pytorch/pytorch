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
  GradBucket(std::vector<at::Tensor> tensors);
  std::vector<at::Tensor> tensors_;
};

using torch::jit::Future;

class CommHookInterface {
 public:
  virtual std::shared_ptr<Future> operate(const GradBucket& bucket);
};

// class CppCommHook : public CommHookInterface {
//  public:
//   CppCommHook(
//       py::object state,
//       std::function<Future(std::vector<at::Tensor>)>& hook);
//   std::shared_ptr<Future> operate(const GradBucket& bucket) override;

//  private:
//   py::object state_;
//   std::function<Future(std::vector<at::Tensor>)> hook_;
// };

class PythonCommHook : public CommHookInterface {
 public:
  PythonCommHook(py::object state, py::object hook);

  std::shared_ptr<Future> operate(const GradBucket& bucket) override;

 private:
  py::object state_;
  py::object hook_;
};
} // namespace c10d

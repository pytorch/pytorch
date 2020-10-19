#pragma once

#include <functional>

#include <ATen/ATen.h>
#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/utils/pybind.h>

namespace c10d {

// Broadcast many tensors to all processes in the process group.
void broadcast_coalesced(
    std::shared_ptr<c10d::ProcessGroup> process_group,
    at::TensorList tensors,
    size_t buffer_size,
    int rank = 0);

// This class passes bucket contents tensor (for multiple replicas) to
// DDP communication hook.
// Optionally in the future this can be enhanced with parameter to bucket
// mappings as well.
class GradBucket {
 public:
  explicit GradBucket(const std::vector<at::Tensor>& tensors)
      : tensors_(tensors) {}
  // Each tensor in the list that getTensors returns refers to the replica on
  // each device. There will be multiple replicas only in the case of single
  // process multiple device mode. In the single process single device mode,
  // this list would consist of only a single tensor.
  const std::vector<at::Tensor>& getTensors() const {
    return tensors_;
  }

 private:
  std::vector<at::Tensor> tensors_;
};

// Base class of both `PythonCommHook` and `CppCommHook`.
// Requires implementing `runHook` method thhat communicate gradients
// asynchronously.
class TORCH_API CommHookInterface {
 public:
  virtual ~CommHookInterface() {}

  // Runs the registered communication hook to communicate gradients
  // asynchronously, Returns a future that holds the communication results.
  virtual c10::intrusive_ptr<torch::jit::Future> runHook(
      GradBucket& bucket) = 0;

  // Returns the resulting tensors once the communication hook result is ready.
  std::vector<at::Tensor> parseFromHookResult(const c10::IValue& result);
};

class TORCH_PYTHON_API PythonCommHook : public CommHookInterface {
 public:
  PythonCommHook(py::object state, py::object hook)
      : state_(std::move(state)), hook_(std::move(hook)) {}

  ~PythonCommHook() override {
    py::gil_scoped_acquire ag;
    state_.dec_ref();
    hook_.dec_ref();
    // Explicitly set state_ and hook_ to nullptr to prevent py::object's dtor
    // to decref on the PyObject again.
    // See Note [Destructing py::object] in python_ivalue.h
    state_.ptr() = nullptr;
    hook_.ptr() = nullptr;
  }

  c10::intrusive_ptr<torch::jit::Future> runHook(GradBucket& bucket) override;

 private:
  // Only needed for stateful communication.
  py::object state_;
  py::object hook_;
};

class TORCH_API CppCommHook : public CommHookInterface {
 public:
  explicit CppCommHook(
      std::function<c10::intrusive_ptr<
          torch::jit::Future>(ProcessGroup*, GradBucket&)>& hook,
      ProcessGroup* process_group = nullptr)
      : process_group_(process_group), hook_(std::move(hook)) {}

  c10::intrusive_ptr<torch::jit::Future> runHook(GradBucket& bucket) override {
    return hook_(process_group_, bucket);
  }

 private:
  // This can be a more generic state if needed.
  ProcessGroup* process_group_; // Not owned.
  std::function<c10::intrusive_ptr<torch::jit::Future>(
      ProcessGroup* process_group,
      GradBucket&)>
      hook_;
};

} // namespace c10d

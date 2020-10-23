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
// Requires implementing 1) `runHook` method that communicates gradients
// asynchronously, and 2) `parseHookResult` method that converts the hook result
// into a tensor vector.
class TORCH_PYTHON_API CommHookInterface {
 public:
  virtual ~CommHookInterface() {}

  // Passes the input grad bucket to the registered communication hook.
  // Once the tensors in the bucket are ready, kicks off the hook asynchronously
  // and returns a future that holds the communication results.
  virtual c10::intrusive_ptr<torch::jit::Future> runHook(
      const GradBucket& bucket) = 0;

  // Returns the resulting tensors once the communication hook result is ready.
  // The resulting tensors will then be copied to the grads of individual
  // parameters.
  virtual std::vector<at::Tensor> parseHookResult(
      const c10::IValue& result) = 0;
};

class TORCH_PYTHON_API PythonCommHook : public CommHookInterface {
 public:
  // Takes a state and a callable hook. The inputs are Python objects.
  // The state is passed to the hook in runHook method, and it can be used to
  // maintain and update any state information during the execution of the hook.
  // The hook performs user-specified processing and returns a future indicating
  // asychronous communication of gradients.
  PythonCommHook(py::object state, py::object hook)
      : state_(std::move(state)), hook_(std::move(hook)) {}

  // The implementation cannot be moved to cpp file, and otherwise it cannot be
  // compiled on Windows platform. This is because the constructor/destructor
  // of a TORCH_API class should only be used in libtorch_core, but this file
  // belongs to libtorch_python_sources.
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

  c10::intrusive_ptr<torch::jit::Future> runHook(
      const GradBucket& bucket) override;

  std::vector<at::Tensor> parseHookResult(const c10::IValue& result) override;

 private:
  // Only needed for stateful communication.
  py::object state_;
  // Indicates an asynchrounous communication of gradients.
  py::object hook_;
};

} // namespace c10d

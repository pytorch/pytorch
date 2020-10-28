#pragma once

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

  std::vector<at::Tensor>& getTensorsRef() {
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
      GradBucket& bucket) = 0;

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

  ~PythonCommHook() override;

  c10::intrusive_ptr<torch::jit::Future> runHook(GradBucket& bucket) override;

  std::vector<at::Tensor> parseHookResult(const c10::IValue& result) override;

 private:
  // Only needed for stateful communication.
  py::object state_;
  py::object hook_;
};

// This CppCommHook interface only requires implementing runHook method that
// potentially uses a state.
// Still need TORCH_PYTHON_API instead of TORCH_API to support Windows platform.
template <typename T>
class TORCH_PYTHON_API CppCommHookInterface : public CommHookInterface {
 public:
  explicit CppCommHookInterface(T& state) : state_(state) {}

  virtual ~CppCommHookInterface() {}

  std::vector<at::Tensor> parseHookResult(const c10::IValue& result) override {
    TORCH_INTERNAL_ASSERT(
        result.isTensor() || result.isTensorList(),
        "expected the hook result is either a Tensor or a TensorList");

    if (result.isTensor()) {
      return {result.toTensor()};
    }

    return result.toTensorVector();
  }

 protected:
  T state_; // Not owned.
};

} // namespace c10d

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
    size_t buffer_size, int rank = 0);

// This class passes bucket contents tensor (for multiple replicas) to
// DDP communication hook.
// Optionally in the future this can be enhanced with parameter to bucket
// mappings as well.
class GradBucket {
 public:
  explicit GradBucket(std::vector<at::Tensor> tensors);
  // Each tensor in the list that getTensors returns refers to the replica on
  // each device. There will be multiple replicas only in the case of single
  // process multiple device mode. In the single process single device mode,
  // this list would consist of only a single tensor.
  const std::vector<at::Tensor>& getTensors() const;

 private:
  std::vector<at::Tensor> tensors_;
};

// DDP's c10d reducer allows communication hooks defined as a sub class
// of CommHookInterface. CommHookInterface is an abstract class and can
// be used to implement both Python and CPP hooks.
struct TORCH_API CommHookInterface {
 public:
  virtual ~CommHookInterface() {}

  // runHook takes a GradBucket type bucket and passes the tensors of
  // this grad bucket to hook's callback. This function is called once
  // the bucket is ready. The hook can perform whatever processing is
  // needed and return a Future that will hold the new value of the grad
  // bucket's tensors once ready.
  virtual c10::intrusive_ptr<torch::jit::Future> runHook(
      const GradBucket& bucket) = 0;

  // Once the grad bucket of Future is ready, c10d reducer will call this
  // function to get the resulting tensors of the grad bucket. Then c10d
  // reducer will use these tensors and copy grads to the grads of individual
  // parameters.
  virtual std::vector<at::Tensor> processFuture(c10::IValue future_value) = 0;
};

// PythonCommHook enables registering a python hook to c10d reducer and is a
// sub class of CommHookInterface.
class TORCH_API PythonCommHook : public CommHookInterface {
 public:
  // The constructor takes a state and a callable hook. Inputs are Python
  // objects. The state is passed to the hook in runHook function can be used to
  // maintain and update any state information that users would like to maintain
  // as part of the training process. The hook can perform whatever processing
  // user specified and return a Future indicating completion of any async work.
  PythonCommHook(py::object state, py::object hook);

  ~PythonCommHook() override {
    py::gil_scoped_acquire ag;
    state_.dec_ref();
    hook_.dec_ref();
    // explicitly setting PyObject* state_ and hook_ to nullptr to prevent
    // py::object's dtor to decref on the PyObject again.
    // See Note [Destructing py::object] in python_ivalue.h
    state_.ptr() = nullptr;
    hook_.ptr() = nullptr;
  }

  c10::intrusive_ptr<torch::jit::Future> runHook(
      const GradBucket& bucket) override;

  std::vector<at::Tensor> processFuture(c10::IValue future_value) override;

 private:
  py::object state_;
  py::object hook_;
};

} // namespace c10d

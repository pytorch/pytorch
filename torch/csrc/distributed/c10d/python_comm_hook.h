#pragma once

#include <memory>

#include <ATen/ATen.h>
#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/comm.h>
#include <torch/csrc/utils/pybind.h>

namespace c10d {

// PythonCommHook enables registering a python hook to c10d reducer and is a
// sub class of CommHookInterface.
class TORCH_PYTHON_API PythonCommHook : public CommHookInterface {
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

}  // namespace c10d

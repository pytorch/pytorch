#pragma once

#include <torch/csrc/distributed/c10d/comm.hpp>

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/utils/pybind.h>

namespace c10d {

class TORCH_PYTHON_API PythonCommHook : public CommHookInterface {
 public:
  // Takes a state and a callable hook. The inputs are Python objects.
  // The state is passed to the hook in runHook method, and it can be used to
  // maintain and update any state information during the execution of the hook.
  // The hook performs user-specified processing and returns a future indicating
  // asynchronous communication of gradients.
  PythonCommHook(py::object state, py::object hook)
      : state_(std::move(state)), hook_(std::move(hook)) {}

  ~PythonCommHook() override;

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;

  at::Tensor parseHookResult(const c10::IValue& result) override;

 private:
  // Only needed for stateful communication.
  py::object state_;
  py::object hook_;
};

} // namespace c10d

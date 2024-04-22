#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/ATen.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <c10/core/SafePyObject.h>

namespace torch::impl::dispatch {
void initDispatchBindings(PyObject* module);

void python_op_registration_trampoline_impl(
    const c10::OperatorHandle& op,
    c10::DispatchKey key,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack,
    bool with_keyset);

class PythonKernelHolder : public c10::OperatorKernel {
  c10::SafePyObject func_;
  c10::DispatchKey dispatch_key_;
  // If "with_keyset", then we expect a keyset as the first arg.
  bool with_keyset_;

 public:
  PythonKernelHolder(
      py::object func,
      c10::DispatchKey dispatch_key,
      bool with_keyset = false);
  void operator()(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack);
};

} // namespace torch::impl::dispatch

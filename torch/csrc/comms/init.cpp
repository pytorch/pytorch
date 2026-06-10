// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/comms.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/comms/gloo/TorchCommGloo.hpp>
#if defined(USE_CUDA) && defined(USE_NCCL) && !defined(USE_ROCM)
#include <torch/csrc/comms/nccl/TorchCommNCCL.hpp>
#endif

// Binding entry points (defined in TorchCommPy.cpp / *Py.cpp). They live in the
// global namespace, matching the translation units that define them.
void init_comms_bindings(py::module_& m);
void init_comms_gloo_bindings(py::module_& m);
#if defined(USE_CUDA) && defined(USE_NCCL) && !defined(USE_ROCM)
void init_comms_nccl_bindings(py::module_& m);
#endif

namespace torch::comms {

namespace {

// Creates the torch._C._comms submodule, populates the core + backend bindings,
// and registers the in-tree backends with the global TorchCommFactory.
//
// Backends are registered here with explicit calls (rather than via static
// initializers) so registration cannot be pruned across the
// torch_python/torch_cuda link boundary, and so gloo/nccl are always available
// after `import torch` without the dlopen-based backend loader.
PyObject* comms_init(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS

  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    throw python_error();
  }
  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();

  auto m = torch_C_m.def_submodule("_comms", "torch.comms C++ bindings");
  init_comms_bindings(m);

  auto gloo_m = m.def_submodule("_comms_gloo", "Gloo backend bindings");
  init_comms_gloo_bindings(gloo_m);
  register_gloo_backend();

#if defined(USE_CUDA) && defined(USE_NCCL) && !defined(USE_ROCM)
  auto nccl_m = m.def_submodule("_comms_nccl", "NCCL backend bindings");
  init_comms_nccl_bindings(nccl_m);
  register_nccl_backend();
#endif

  Py_RETURN_TRUE;
  END_HANDLE_TH_ERRORS
}

} // namespace

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
static PyMethodDef methods[] = { // NOLINT
    {"_comms_init", comms_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace torch::comms

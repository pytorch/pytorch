#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAGreenContext.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

// Cargo culted partially from csrc/cuda/Stream.cpp

void THCPGreenContext_init(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<at::cuda::GreenContext>(m, "_CUDAGreenContext")
      .def_static("create", &::at::cuda::GreenContext::create)
      .def("make_current", &::at::cuda::GreenContext::makeCurrent)
      .def("pop_current", &::at::cuda::GreenContext::popCurrent);
}

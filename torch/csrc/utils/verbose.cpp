#include <ATen/native/verbose_wrapper.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {

void initVerboseBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto verbose = m.def_submodule("_verbose", "MKL, MKLDNN verbose");
  verbose.def("mkl_set_verbose", torch::verbose::_mkl_set_verbose);
  verbose.def("mkldnn_set_verbose", torch::verbose::_mkldnn_set_verbose);
}

} // namespace torch

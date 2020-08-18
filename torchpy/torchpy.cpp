#include <torchpy.h>
#include <assert.h>
#include <stdio.h>
#include <torch/csrc/jit/python/pybind_utils.h>
// #include <torch/script.h>
// #include <torch/torch.h>
#include <pybind11/embed.h>
#include <iostream>
// #include "torch/csrc/autograd/python_variable.h"

// using namespace pybind11::literals; // to bring in the `_a` literal

namespace torchpy {

void init() {
  py::initialize_interpreter();
  // Make sure torch is loaded before anything else
  py::module::import("torch");

  // Tensor is used directly in the loaded pt code
  // Why does importing it here not obviate the need to import it in loader.py?
  // py::exec("from torch import Tensor");

  // Make loader importable
  py::exec(R"(
        import sys
        sys.path.append('torchpy')
    )");
}

void finalize() {
  py::finalize_interpreter();
}
const PyModule load(const char* filename) {
  std::cout << "load()" << std::endl;

  // auto simple_loader = py::module::import("simple_loader");
  auto loader = py::module::import("loader");
  auto load = loader.attr("load");

  std::cout << "callobject load" << std::endl;
  auto model = load(filename);
  auto mod = PyModule(model);
  return mod;
}

PyModule::PyModule(py::object model) : _model(model) {}
// make sure not to destroy python objects in here as
// py finalize may have happened before these modules
// get destroyed.
PyModule::~PyModule() {
}

at::Tensor PyModule::forward(at::Tensor input) {
  py::object forward = _model.attr("forward");
  py::object py_output = forward(input);
  at::Tensor output = py::cast<at::Tensor>(py_output);
  return output;
}
} // namespace torchpy
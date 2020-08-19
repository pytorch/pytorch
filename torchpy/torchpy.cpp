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

PyThreadState* mainThreadState = NULL;

/*
//in main thread

Py_Initialize();
PyEval_InitThreads();
mainThreadState = PyThreadState_Get();
PyEval_ReleaseLock();

//in threaded thread
PyEval_AcquireLock();
PyInterpreterState * mainInterpreterState = mainThreadState->interp;
PyThreadState * myThreadState = PyThreadState_New(mainInterpreterState);
PyEval_ReleaseLock();

 * embeded python part
 * PyEval_CallObject() for example
 */

void init() {
  Py_Initialize();
  // py::initialize_interpreter();
  // Make sure torch is loaded before anything else
  py::module::import("torch");

  // Why does importing it here not obviate the need to import it in loader.py?
  // py::exec("from torch import Tensor");

  // Make loader importable
  py::exec(R"(
        import sys
        sys.path.append('torchpy')
    )");

  // Enable other threads to use the interpreter
  assert(PyEval_ThreadsInitialized() != 0);
}

void finalize() {
  Py_Finalize();
  // py::finalize_interpreter();
}
const PyModule load(const char* filename) {
  std::cout << "load()" << std::endl;

  auto loader = py::module::import("loader");
  auto load = loader.attr("load");

  std::cout << "callobject load" << std::endl;
  auto model = load(filename);
  auto mod = PyModule(model);

  mainThreadState = PyEval_SaveThread();
  std::cout << "load return" << std::endl;

  return mod;
}

PyModule::PyModule(py::object model) : _model(model) {}
// make sure not to destroy python objects in here as
// py finalize may have happened before these modules
// get destroyed.
PyModule::~PyModule() {
}

at::Tensor PyModule::forward(at::Tensor input) {
  std::cout << "forward" << std::endl;
  PyEval_RestoreThread(mainThreadState);

  std::cout << "restored" << std::endl;
  py::object forward = _model.attr("forward");
  std::cout << "called forward" << std::endl;
  py::object py_output = forward(input);
  std::cout << "casting output" << std::endl;
  at::Tensor output = py::cast<at::Tensor>(py_output);
  std::cout << "returning output" << std::endl;
  return output;
}
} // namespace torchpy
#include <torchpy.h>
#include <assert.h>
#include <pybind11/embed.h>
#include <stdio.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <iostream>
#include <mutex>
#include <thread>

namespace torchpy {

PyThreadState* mainThreadState = NULL;

void init() {
  Py_Initialize();
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
  assert(PyGILState_Check() == 1);
  assert(PyEval_ThreadsInitialized() != 0);
  mainThreadState = PyEval_SaveThread(); // save our state, release GIL
}

void finalize() {
  PyEval_RestoreThread(mainThreadState); // Acquire GIL, resume our state
  Py_Finalize();
}
const PyModule load(const char* filename) {
  PyEval_RestoreThread(mainThreadState); // Acquire GIL, resume our state
  assert(PyGILState_Check() == 1);

  auto loader = py::module::import("loader");
  auto load = loader.attr("load");

  auto model = load(filename);
  auto forward = model.attr("forward");
  auto pymodule = PyModule(forward);

  mainThreadState = PyEval_SaveThread(); // save our state, release GIL
  return pymodule;
}

PyModule::PyModule(py::object model_forward) : _model_forward(model_forward) {}

PyModule::~PyModule() {
  PyGILState_STATE gil_state = PyGILState_Ensure();
  { _model_forward.dec_ref(); }
  PyGILState_Release(gil_state);
  assert(_thread_states.empty());
}

void PyModule::thread_begin() {
  std::thread::id this_id = std::this_thread::get_id();
  PyInterpreterState* mainInterpreterState = mainThreadState->interp;
  PyThreadState* myThreadState =
      PyThreadState_New(mainInterpreterState); // GIL not needed
  PyEval_RestoreThread(myThreadState); // Acquires GIL
  _thread_states[this_id] = myThreadState;
}

void PyModule::thread_end() {
  std::thread::id this_id = std::this_thread::get_id();
  PyThreadState* myThreadState = _thread_states[this_id];
  _thread_states.erase(this_id);
  PyThreadState_Clear(myThreadState); // GIL needed
  PyEval_ReleaseThread(myThreadState); // Releases GIL
  PyThreadState_Delete(myThreadState); // GIL not needed
}

at::Tensor PyModule::forward(at::Tensor input) {
  at::Tensor output;
  PyGILState_STATE gil_state = PyGILState_Ensure();
  {
    py::object py_output = _model_forward(input);
    // TODO is this going to leak?
    // added it to prevent crash wehn using 'output' tensor in callee of
    // forward()
    py_output.inc_ref();
    output = py::cast<at::Tensor>(py_output);
  }
  PyGILState_Release(gil_state);

  return output;
}
} // namespace torchpy
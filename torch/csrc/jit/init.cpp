#include <Python.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "THP.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/graph_fuser.h"
#include "torch/csrc/jit/init_pass.h"
#include "torch/csrc/jit/dead_code_elimination.h"
#include "torch/csrc/jit/python_tracer.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/toffee/export.h"

PyObject * THPJIT_initExtension(PyObject *_unused)
{
  // Leaving this code here, because it will likely be useful at some point
  //PyObject *jit_module = PyImport_ImportModule("torch.jit");
  //THPUtils_assert(jit_module, "class loader couldn't access "
          //"torch.jit module");
  //PyObject *jit_dict = PyModule_GetDict(jit_module);

  Py_RETURN_TRUE;
}

// stub to run all C++ only tests for the JIT
// the stuff in test_jit.cpp is kept separate from the rest of PyTorch
// so we can build and iterate on it faster.
// from test_jit.cpp
namespace torch  { namespace jit { extern void runJITCPPTests(); } };

namespace {

using namespace torch::jit;

using pass_type = void (std::shared_ptr<Graph>&);

template<pass_type pass>
PyObject * wrap_pass(PyObject *_unused, PyObject *py_state) {
  HANDLE_TH_ERRORS
  auto trace = py::handle(py_state).cast<tracer::TracingState*>();
  pass(trace->graph);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * run_cpp_tests(PyObject *_unused, PyObject *_unused2) {
    HANDLE_TH_ERRORS
    runJITCPPTests();
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

struct PyMethodDef _THPJIT_methods[] = {
  {"_jit_init",       (PyCFunction)THPJIT_initExtension,      METH_NOARGS,  NULL},
  {"_jit_pass_init", (PyCFunction)wrap_pass<MatchJITOps>,     METH_O,       "init"},
  {"_jit_pass_fuse", (PyCFunction)wrap_pass<FuseGraph>,       METH_O,       "fuse"},
  {"_jit_pass_dce",  (PyCFunction)wrap_pass<EliminateDeadCode>, METH_O,     "dce"},
  {"_jit_pass_lint", (PyCFunction)wrap_pass<LintGraph>,       METH_O,       "lint"},
  {"_jit_run_cpp_tests",(PyCFunction)run_cpp_tests,           METH_NOARGS,  NULL},
  {NULL}
};

} // anonymous namespace

PyMethodDef* THPJIT_methods() {
    return _THPJIT_methods;
}

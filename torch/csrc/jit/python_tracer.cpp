#include <Python.h>
#include "torch/csrc/jit/python_tracer.h"

#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/python_ir.h"
#include "torch/csrc/THP.h"

using namespace torch::autograd;
using namespace torch::jit;

PyObject * THPTracer_enter(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* input_objs;
  GlobalTracingState.enter();
  if (!PyArg_ParseTuple(args, "O", &input_objs)) {
    return NULL;
  }
  auto & graph = GlobalTracingState.current();
  THPUtils_assert(PyTuple_Check(input_objs), "inputs argument is "
    "expected to be a tuple, but got %s", THPUtils_typename(input_objs));
  Py_ssize_t num_inputs = PyTuple_GET_SIZE(input_objs);

  for (int i = 0; i < num_inputs; i++) {
    PyObject* input_obj = PyTuple_GET_ITEM(input_objs, i);
    THPUtils_assert(THPVariable_Check(input_obj), "element %d of input "
        "tuple is not a Variable", i);
    ((THPVariable*)input_obj)->cdata->trace_value = graph.addInput(graph.create<Param>());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPTracer_exit(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* output_objs;
  if (!PyArg_ParseTuple(args, "O", &output_objs)) {
    return NULL;
  }
  auto graph = GlobalTracingState.exit();

  THPUtils_assert(PyTuple_Check(output_objs), "outputs argument is "
    "expected to be a tuple, but got %s", THPUtils_typename(output_objs));
  Py_ssize_t num_outputs = PyTuple_GET_SIZE(output_objs);

  for (int i = 0; i < num_outputs; i++) {
    PyObject* output_obj = PyTuple_GET_ITEM(output_objs, i);
    THPUtils_assert(THPVariable_Check(output_obj), "element %d of outputs "
        "tuple is not a Variable", i);
    THPUtils_assert(((THPVariable*)output_obj)->cdata->trace_value, "element %d of outputs "
        "was not traced", i)
    graph->addOutput(((THPVariable*)output_obj)->cdata->trace_value);
  }

  return THPGraph_Wrap(std::move(graph));
  END_HANDLE_TH_ERRORS
}

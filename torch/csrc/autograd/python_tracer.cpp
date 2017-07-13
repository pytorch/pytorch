#include <Python.h>
#include "torch/csrc/autograd/python_tracer.h"

#include "torch/csrc/autograd/tracer.h"
#include "torch/csrc/THP.h"

using namespace torch::autograd;

PyObject * THPTracer_enter(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* input_objs;
  Tracer_enter();
  if (!PyArg_ParseTuple(args, "O", &input_objs)) {
    return NULL;
  }

  THPUtils_assert(PyTuple_Check(input_objs), "inputs argument is "
    "expected to be a tuple, but got %s", THPUtils_typename(input_objs));
  Py_ssize_t num_inputs = PyTuple_GET_SIZE(input_objs);

  for (int i = 0; i < num_inputs; i++) {
    PyObject* input_obj = PyTuple_GET_ITEM(input_objs, i);
    THPUtils_assert(THPVariable_Check(input_obj), "element %d of input "
        "tuple is not a Variable", i);

    auto local = GlobalTracingState->makeValue(nullptr,0);
    ((THPVariable*)input_obj)->cdata->trace_value = ValueRef(local.get());
    GlobalTracingState->graph->inputs.push_back(std::move(local));
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

  THPUtils_assert(PyTuple_Check(output_objs), "outputs argument is "
    "expected to be a tuple, but got %s", THPUtils_typename(output_objs));
  Py_ssize_t num_outputs = PyTuple_GET_SIZE(output_objs);

  value_list locals;
  locals.reserve(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    PyObject* output_obj = PyTuple_GET_ITEM(output_objs, i);
    THPUtils_assert(THPVariable_Check(output_obj), "element %d of outputs "
        "tuple is not a Variable", i);
    THPUtils_assert(((THPVariable*)output_obj)->cdata->trace_value, "element %d of outputs "
        "was not traced", i)
    locals.push_back(((THPVariable*)output_obj)->cdata->trace_value);
  }

  return THPGraph_Wrap(Tracer_exit(locals));
  END_HANDLE_TH_ERRORS
}

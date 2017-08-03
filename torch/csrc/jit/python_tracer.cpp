#include <Python.h>
#include "torch/csrc/jit/python_tracer.h"

#include "torch/csrc/autograd/jit_closure.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/assert.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/THP.h"

#include <sstream>

using namespace torch::autograd;
using namespace torch::jit;

// TODO: THIS IMPLEMENTATION CURRENTLY LEAKS IF STORED PYTHON OBJECTS IN AST
// HAVE BACK REFERENCES, DUE TO CYCLE.  Need to fix this at some point.

static int THPTracingState_traverse(THPTracingState *self, visitproc visit, void *arg)
{
  return 0; // LEAK!
}

static int THPTracingState_clear(THPTracingState *self)
{
  return 0; // LEAK! if implemented, must also implement traverse
}

static void THPTracingState_dealloc(THPTracingState* self)
{
  PyObject_GC_UnTrack(self);
  JIT_ASSERT(self->cdata);
  self->cdata.~shared_ptr<tracer::TracingState>();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

typedef PyObject *(*getter)(PyObject *, void *);

static struct PyGetSetDef THPTracingState_properties[] = {
  {NULL}
};

static PyObject* THPTracingState_str(THPTracingState *self) {
  std::stringstream ss;
  ss << *self->cdata->graph;
  return THPUtils_packString(ss.str());
}

PyTypeObject THPTracingStateType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C.TracingState",                 /* tp_name */
  sizeof(THPTracingState),                      /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPTracingState_dealloc,          /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  (reprfunc)THPTracingState_str,                /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  NULL,                                  /* tp_doc */
  (traverseproc)THPTracingState_traverse,       /* tp_traverse */
  (inquiry)THPTracingState_clear,               /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  THPTracingState_properties,                   /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  // TODO: add me, seems reasonable
  0                                      /* tp_new */
};

PyObject* THPTracingState_Wrap(std::shared_ptr<tracer::TracingState> e)
{
  if (!e) {
    Py_RETURN_NONE;
  } else {
    THPTracingState* obj = (THPTracingState*)THPTracingStateType.tp_alloc(&THPTracingStateType, 0);
    if (!obj) return nullptr;
    new (&obj->cdata) std::shared_ptr<tracer::TracingState>(e);
    return (PyObject*) obj;
  }
}

bool THPTracingState_Check(PyObject *obj) {
  return PyObject_IsInstance(obj, (PyObject*)&THPTracingStateType);
}

bool THPTracer_initModule(PyObject *module) {
  if (PyType_Ready(&THPTracingStateType) < 0)
    return false;
  Py_INCREF(&THPTracingStateType);
  PyModule_AddObject(module, "TracingState", (PyObject *)&THPTracingStateType);
  return true;
}

PyObject * THPTracer_enter(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* input_objs;
  if (!PyArg_ParseTuple(args, "O", &input_objs)) {
    return NULL;
  }
  THPUtils_assert(PyTuple_Check(input_objs), "inputs argument is "
    "expected to be a tuple, but got %s", THPUtils_typename(input_objs));
  Py_ssize_t num_inputs = PyTuple_GET_SIZE(input_objs);

  variable_list inputs;
  for (int i = 0; i < num_inputs; i++) {
    PyObject* input_obj = PyTuple_GET_ITEM(input_objs, i);
    THPUtils_assert(THPVariable_Check(input_obj), "element %d of input "
        "tuple is not a Variable", i);
    inputs.emplace_back(((THPVariable*)input_obj)->cdata);
  }

  THPObjectPtr tracing_state {THPTracingState_Wrap(tracer::enter(inputs))};
  THPObjectPtr new_inputs {PyTuple_New(num_inputs)};
  for (int i = 0; i < num_inputs; ++i) {
    PyTuple_SET_ITEM(new_inputs.get(), i, THPVariable_Wrap(inputs[i]));
  }
  return Py_BuildValue("OO", tracing_state.release(), new_inputs.release());
  END_HANDLE_TH_ERRORS
}

PyObject * THPTracer_exit(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* output_objs = NULL;
  if (!PyArg_ParseTuple(args, "O", &output_objs)) {
    return NULL;
  }

  THPUtils_assert(PyTuple_Check(output_objs), "outputs argument is "
    "expected to be a tuple, but got %s", THPUtils_typename(output_objs));
  Py_ssize_t num_outputs = PyTuple_GET_SIZE(output_objs);

  variable_list outputs;
  for (int i = 0; i < num_outputs; i++) {
    PyObject* output_obj = PyTuple_GET_ITEM(output_objs, i);
    THPUtils_assert(THPVariable_Check(output_obj), "element %d of outputs "
        "tuple is not a Variable", i);
    auto& var = ((THPVariable*)output_obj)->cdata;
    outputs.emplace_back(var);
  }

  // TODO: reset output vars
  tracer::exit(outputs);

  THPObjectPtr new_outputs(PyTuple_New(num_outputs));
  for (int i = 0; i < num_outputs; ++i) {
    PyTuple_SET_ITEM(new_outputs.get(), i, THPVariable_Wrap(outputs[i]));
  }
  return new_outputs.release();
  END_HANDLE_TH_ERRORS
}

PyObject * THPTracer_createAutogradClosure(PyObject *_unused, PyObject *pystate) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPTracingState_Check(pystate), "getClosure expected a TracingState, but got %s",
      THPUtils_typename(pystate));
  auto& state = ((THPTracingState*)pystate)->cdata;

  auto closure = createAutogradClosure(state->graph.get());

  return THPWrapper_New(closure.release(),
                        [](void *fn_list) { delete reinterpret_cast<AutogradClosure*>(fn_list); });
  END_HANDLE_TH_ERRORS
}

#include "torch/csrc/jit/python_ir.h"
#include "torch/csrc/utils/python_strings.h"

#include <sstream>

using namespace torch::autograd;

// TODO: THIS IMPLEMENTATION CURRENTLY LEAKS IF STORED PYTHON OBJECTS IN AST
// HAVE BACK REFERENCES, DUE TO CYCLE.  Need to fix this at some point.

PyObject* THPGraphClass = nullptr;

PyObject* THPGraph_Wrap(std::unique_ptr<torch::jit::Graph> e)
{
  if (!e) {
    Py_RETURN_NONE;
  } else {
    auto type = (PyTypeObject*) THPGraphClass;
    THPGraph* obj = (THPGraph*)type->tp_alloc(type, 0);
    if (obj) {
      obj->cdata = e.release();
    }
    return (PyObject*) obj;
  }
}

static int THPGraph_traverse(THPGraph *self, visitproc visit, void *arg)
{
  return 0; // LEAK!
}

static int THPGraph_clear(THPGraph *self)
{
  return 0; // LEAK! if implemented, must also implement traverse
}

static void THPGraph_dealloc(THPGraph* self)
{
  PyObject_GC_UnTrack(self);
  assert(self->cdata);
  delete self->cdata;
  Py_TYPE(self)->tp_free((PyObject*)self);
}

typedef PyObject *(*getter)(PyObject *, void *);

/*
PyObject *THPNode_get_inputs(THPNode *self) {
  auto& node = self->cdata;
  auto size = node->inputs.size();
  PyObject* pyInputs(PyTuple_New(size));
  if (!pyInputs) Py_RETURN_NONE;
  for (size_t i = 0; i < size; i++) {
    auto input = PyTuple_New(2);
    PyTuple_SET_ITEM(input, 0, THPNode_Wrap(node->inputs[0].node));
    PyTuple_SET_ITEM(input, 1, PyLong_FromLong(node->inputs[0].output_nr));
    PyTuple_SET_ITEM(pyInputs, i, input);
  }
  return pyInputs;
}

PyObject *THPNode_get_name(THPNode *self) {
  auto& node = self->cdata;
  return THPUtils_packString(node->name());
}
*/

static struct PyGetSetDef THPGraph_properties[] = {
//  {"_inputs", (getter)THPNode_get_inputs, NULL, NULL, NULL},
//  {"_name", (getter)THPNode_get_name, NULL, NULL, NULL},
  {NULL}
};

static PyObject* THPGraph_str(THPGraph *self) {
  std::stringstream ss;
  ss << *self->cdata;
  return THPUtils_packString(ss.str());
}

PyTypeObject THPGraphType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._GraphBase",                  /* tp_name */
  sizeof(THPGraph),                       /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPGraph_dealloc,           /* tp_dealloc */
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
  (reprfunc)THPGraph_str,                 /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  NULL,                                  /* tp_doc */
  (traverseproc)THPGraph_traverse,        /* tp_traverse */
  (inquiry)THPGraph_clear,                /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  THPGraph_properties,                    /* tp_getset */
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

bool THPIR_initModule(PyObject *module) {
  if (PyType_Ready(&THPGraphType) < 0)
    return false;
  Py_INCREF(&THPGraphType);
  PyModule_AddObject(module, "_GraphBase", (PyObject *)&THPGraphType);
  return true;
}

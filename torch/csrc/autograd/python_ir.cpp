#include "torch/csrc/autograd/python_ir.h"
#include "torch/csrc/utils/python_strings.h"

using namespace torch::autograd;

PyObject* THPNodeClass = nullptr;

PyObject* THPNode_Wrap(const std::shared_ptr<Node>& node)
{
  if (!node) {
    Py_RETURN_NONE;
  } else {
    auto type = (PyTypeObject*) THPNodeClass;
    THPNode* obj = (THPNode*)type->tp_alloc(type, 0);
    if (obj) {
      obj->cdata = node;
    }
    return (PyObject*) obj;
  }
}

static int THPNode_traverse(THPNode *self, visitproc visit, void *arg)
{
  if (self->cdata) {
    if (auto fn = dynamic_cast<PyNode*>(self->cdata.get())) {
      Py_VISIT(fn->pyobj);
    }
  }
  return 0;
}

static int THPNode_clear(THPNode *self)
{
  if (self->cdata) {
    if (auto fn = dynamic_cast<PyNode*>(self->cdata.get())) {
      fn->pyobj = nullptr;
    }
  }
  self->cdata.reset();
  return 0;
}

static void THPNode_dealloc(THPNode* self)
{
  PyObject_GC_UnTrack(self);
  THPNode_clear(self);
  self->cdata.~shared_ptr<Node>();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

typedef PyObject *(*getter)(PyObject *, void *);

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

static struct PyGetSetDef THPNode_properties[] = {
  {"_inputs", (getter)THPNode_get_inputs, NULL, NULL, NULL},
  {"_name", (getter)THPNode_get_name, NULL, NULL, NULL},
  {NULL}
};

PyTypeObject THPNodeType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._NodeBase",                  /* tp_name */
  sizeof(THPNode),                       /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPNode_dealloc,           /* tp_dealloc */
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
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  NULL,                                  /* tp_doc */
  (traverseproc)THPNode_traverse,        /* tp_traverse */
  (inquiry)THPNode_clear,                /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  THPNode_properties,                    /* tp_getset */
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

bool THPNode_initModule(PyObject *module) {
  if (PyType_Ready(&THPNodeType) < 0)
    return false;
  Py_INCREF(&THPNodeType);
  PyModule_AddObject(module, "_NodeBase", (PyObject *)&THPNodeType);
  return true;
}

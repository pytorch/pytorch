#include <torch/csrc/fx/node.h>

#include <structmember.h>

////////////////////////////////
// NodeBase
///////////////////////////////

struct NodeBase {
  PyObject_HEAD bool _erased;
  NodeBase* _prev;
  NodeBase* _next;
};

static PyObject* NodeBase_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  PyObject* self = type->tp_alloc(type, 0);
  if (!self)
    return nullptr;
  return self;
}

static int NodeBase_init_fn(NodeBase* self, PyObject* args, PyObject* kwds) {
  self->_erased = false;
  self->_prev = self;
  self->_next = self;
  return 0;
}

static void NodeBase_dealloc(NodeBase* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static struct PyMemberDef NodeBase_members[] = {
    {"_erased", T_BOOL, offsetof(NodeBase, _erased), 0, nullptr},
    {"_prev", T_OBJECT, offsetof(NodeBase, _prev), 0, nullptr},
    {"_next", T_OBJECT, offsetof(NodeBase, _next), 0, nullptr},
    {nullptr} /* Sentinel */
};

static PyTypeObject NodeBaseType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C.NodeBase", /* tp_name */
    sizeof(NodeBase), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)NodeBase_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    NodeBase_members, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)NodeBase_init_fn, /* tp_init */
    nullptr, /* tp_alloc */
    NodeBase_new, /* tp_new */
};

bool NodeBase_init(PyObject* module) {
  if (PyType_Ready(&NodeBaseType) < 0) {
    return false;
  }
  Py_INCREF(&NodeBaseType);
  if (PyModule_AddObject(module, "NodeBase", (PyObject*)&NodeBaseType) != 0) {
    return false;
  }
  return true;
}

////////////////////////////////
// NodeIter
///////////////////////////////

struct NodeIter {
  PyObject_HEAD bool _reversed;
  NodeBase* _root;
  NodeBase* _cur;
};

static PyObject* NodeIter_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  PyObject* self = type->tp_alloc(type, 0);
  if (!self)
    return nullptr;
  return self;
}

static int NodeIter_init_fn(NodeIter* self, PyObject* args, PyObject* kwargs) {
  NodeBase* root = nullptr;
  bool reversed = false;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* keywords[] = {"root", "reversed", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "Ob|",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(keywords),
          &root,
          &reversed)) {
    return -1;
  }
  self->_reversed = reversed;
  self->_root = root;
  self->_cur = root;
  return 0;
}

static void NodeIter_dealloc(NodeIter* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* NodeIter_iter(PyObject* self) {
  Py_INCREF(self);
  return self;
}

template <bool reversed>
PyObject* NodeIter_iternext_helper(NodeIter* self) {
  if constexpr (reversed) {
    self->_cur = self->_cur->_prev;
  } else {
    self->_cur = self->_cur->_next;
  }
  while (self->_cur != self->_root) {
    if (!self->_cur->_erased) {
      Py_INCREF(self->_cur);
      return (PyObject*)self->_cur;
    }
    if constexpr (reversed) {
      self->_cur = self->_cur->_prev;
    } else {
      self->_cur = self->_cur->_next;
    }
  }
  PyErr_SetNone(PyExc_StopIteration);
  return nullptr;
}

PyObject* NodeIter_iternext(PyObject* _self) {
  NodeIter* self = (NodeIter*)_self;
  if (self->_reversed) {
    return NodeIter_iternext_helper<true>(self);
  } else {
    return NodeIter_iternext_helper<false>(self);
  }
}

static PyTypeObject NodeIterType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._NodeIter", /* tp_name */
    sizeof(NodeIter), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)NodeIter_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    NodeIter_iter, /* tp_iter */
    NodeIter_iternext, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)NodeIter_init_fn, /* tp_init */
    nullptr, /* tp_alloc */
    NodeIter_new, /* tp_new */
};

bool NodeIter_init(PyObject* module) {
  if (PyType_Ready(&NodeIterType) < 0) {
    return false;
  }
  Py_INCREF(&NodeIterType);
  if (PyModule_AddObject(module, "_NodeIter", (PyObject*)&NodeIterType) != 0) {
    return false;
  }
  return true;
}

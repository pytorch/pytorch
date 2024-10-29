#include <torch/csrc/fx/node.h>

#include <structmember.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

////////////////////////////////
// NodeBase
///////////////////////////////

struct NodeBase {
  PyObject_HEAD
  bool _erased;
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
  Py_INCREF(self);
  self->_prev = self;
  Py_INCREF(self);
  self->_next = self;
  return 0;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static struct PyMemberDef NodeBase_members[] = {
    {"_erased", T_BOOL, offsetof(NodeBase, _erased), 0, nullptr},
    {"_prev", T_OBJECT_EX, offsetof(NodeBase, _prev), 0, nullptr},
    {"_next", T_OBJECT_EX, offsetof(NodeBase, _next), 0, nullptr},
    {nullptr} /* Sentinel */
};

static int NodeBase_traverse(NodeBase* self, visitproc visit, void* arg) {
  Py_VISIT(self->_prev);
  Py_VISIT(self->_next);
  return 0;
}

static int NodeBase_clear(NodeBase* self) {
  Py_CLEAR(self->_prev);
  Py_CLEAR(self->_next);
  return 0;
}

static void NodeBase_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  (void)NodeBase_clear((NodeBase*)self);
  Py_TYPE(self)->tp_free(self);
}

static PyTypeObject NodeBaseType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._NodeBase", /* tp_name */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC, /* tp_flags */
    nullptr, /* tp_doc */
    (traverseproc)NodeBase_traverse, /* tp_traverse */
    (inquiry)NodeBase_clear, /* tp_clear */
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
  if (PyModule_AddType(module, &NodeBaseType) < 0) {
    return false;
  }
  return true;
}

////////////////////////////////
// NodeIter
////////////////////////////////

struct NodeIter {
  PyObject_HEAD
  bool _reversed;
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
  Py_INCREF(root);
  self->_root = root;
  Py_INCREF(root);
  self->_cur = root;
  return 0;
}

template <bool reversed>
PyObject* NodeIter_iternext_helper(NodeIter* self) {
  // It should be possible to relax the ref counting here
  // but in practice, we do not have that many _erased Nodes,
  // so probably not worth it.
  if constexpr (reversed) {
    NodeBase* prev = (NodeBase*)Py_NewRef(self->_cur->_prev);
    Py_CLEAR(self->_cur);
    self->_cur = prev;
  } else {
    NodeBase* next = (NodeBase*)Py_NewRef(self->_cur->_next);
    Py_CLEAR(self->_cur);
    self->_cur = next;
  }
  while (self->_cur != self->_root) {
    if (!self->_cur->_erased) {
      Py_INCREF(self->_cur);
      return (PyObject*)self->_cur;
    }
    if constexpr (reversed) {
      NodeBase* prev = (NodeBase*)Py_NewRef(self->_cur->_prev);
      Py_CLEAR(self->_cur);
      self->_cur = prev;
    } else {
      NodeBase* next = (NodeBase*)Py_NewRef(self->_cur->_next);
      Py_CLEAR(self->_cur);
      self->_cur = next;
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

static int NodeIter_traverse(NodeIter* self, visitproc visit, void* arg) {
  Py_VISIT(self->_root);
  Py_VISIT(self->_cur);
  return 0;
}

static int NodeIter_clear(NodeIter* self) {
  Py_CLEAR(self->_root);
  Py_CLEAR(self->_cur);
  return 0;
}

static void NodeIter_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  (void)NodeIter_clear((NodeIter*)self);
  Py_TYPE(self)->tp_free(self);
}

static PyTypeObject NodeIterType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._NodeIter", /* tp_name */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /* tp_flags */
    nullptr, /* tp_doc */
    (traverseproc)NodeIter_traverse, /* tp_traverse */
    (inquiry)NodeIter_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    PyObject_SelfIter, /* tp_iter */
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
  if (PyModule_AddType(module, &NodeIterType) < 0) {
    return false;
  }
  return true;
}

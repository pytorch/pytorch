#include <torch/csrc/fx/node.h>

#include <structmember.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

namespace {

struct NodeBase;

// Thrown to exit out of a C++ function and return an error to Python.
class PythonError : public std::exception {};

inline static PyObject* import_from(const char* module_name, const char* name) {
  THPObjectPtr module(PyImport_ImportModule(module_name));
  if (!module) {
    throw PythonError();
  }
  PyObject* result = PyObject_GetAttrString(module, name);
  if (!result) {
    throw PythonError();
  }
  return result;
}

inline static PyObject* immutable_list_cls() {
  static PyObject* immutable_list_cls = nullptr;
  if (!immutable_list_cls) {
    immutable_list_cls =
        import_from("torch.fx.immutable_collections", "immutable_list");
  }
  return immutable_list_cls;
}

inline static PyObject* immutable_dict_cls() {
  static PyObject* immutable_dict_cls = nullptr;
  if (!immutable_dict_cls) {
    immutable_dict_cls =
        import_from("torch.fx.immutable_collections", "immutable_dict");
  }
  return immutable_dict_cls;
}

inline static bool is_node(PyObject* obj) {
  static PyObject* node_cls = nullptr;
  if (!node_cls) {
    node_cls = import_from("torch.fx.node", "Node");
  }
  return PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(node_cls));
}

inline static bool exact_type(PyObject* obj, PyObject* typ) {
  return Py_TYPE(obj) == reinterpret_cast<PyTypeObject*>(typ);
}

template <typename F>
inline static PyObject* map_aggregate(PyObject* a, F fn) {
  // Invariant: this function will throw an exception and never return nullptr.
  // Case 1: a is a tuple.
  if (PyTuple_Check(a)) {
    Py_ssize_t n = PyTuple_GET_SIZE(a);
    if (n == 0 && PyTuple_CheckExact(a)) {
      return Py_NewRef(a);
    }
    THPObjectPtr new_tuple(PyTuple_New(n));
    if (!new_tuple) {
      throw PythonError();
    }
    for (Py_ssize_t i = 0; i < n; i++) {
      PyObject* elem = PyTuple_GET_ITEM(a, i); // Borrowed reference.
      // PyTuple_SET_ITEM steals reference to result of map_aggregate
      PyTuple_SET_ITEM(new_tuple.get(), i, map_aggregate(elem, fn));
    }
    // If the tuple has a "_fields" attribute, assume it is a NamedTuple.
    if (!PyTuple_CheckExact(a) && PyObject_HasAttrString(a, "_fields")) {
      // Call type_obj with new_tuple as arguments (i.e. type(a)(*new_tuple))
      return PyObject_CallObject(
          reinterpret_cast<PyObject*>(Py_TYPE(a)), new_tuple);
    } else {
      return new_tuple.release();
    }
  }
  // Case 2: a is a list.
  else if (PyList_Check(a)) {
    Py_ssize_t n = PyList_GET_SIZE(a);
    if (n == 0 && exact_type(a, immutable_list_cls())) {
      return Py_NewRef(a);
    }
    THPObjectPtr result(PyObject_CallNoArgs(immutable_list_cls()));
    if (!result) {
      throw PythonError();
    }
    for (Py_ssize_t i = 0; i < n; i++) {
      PyObject* elem = PyList_GET_ITEM(a, i); // borrowed ref
      THPObjectPtr mapped(map_aggregate(elem, fn));
      if (PyList_Append(result.get(), mapped.get()) < 0) {
        throw PythonError();
      }
    }
    return result.release();
  }
  // Case 3: a is a dict.
  else if (PyDict_Check(a)) {
    if (PyDict_GET_SIZE(a) == 0 && exact_type(a, immutable_dict_cls())) {
      return Py_NewRef(a);
    }
    THPObjectPtr result(PyObject_CallNoArgs(immutable_dict_cls()));
    if (!result) {
      throw PythonError();
    }
    PyObject *key = nullptr, *value = nullptr; // borrowed
    Py_ssize_t pos = 0;
    while (PyDict_Next(a, &pos, &key, &value)) {
      THPObjectPtr mapped(map_aggregate(value, fn));
      if (PyDict_SetItem(result.get(), key, mapped.get()) < 0) {
        throw PythonError();
      }
    }
    return result.release();
  }
  // Case 4: a is a slice.
  else if (PySlice_Check(a)) {
    // Get start, stop, and step attributes.
    THPObjectPtr start(PyObject_GetAttrString(a, "start"));
    THPObjectPtr stop(PyObject_GetAttrString(a, "stop"));
    THPObjectPtr step(PyObject_GetAttrString(a, "step"));
    if (!start || !stop || !step) {
      throw PythonError();
    }
    THPObjectPtr mapped_start(map_aggregate(start, fn));
    THPObjectPtr mapped_stop(map_aggregate(stop, fn));
    THPObjectPtr mapped_step(map_aggregate(step, fn));
    return PySlice_New(
        mapped_start.get(), mapped_stop.get(), mapped_step.get());
  }
  // Default case: call fn(a).
  else {
    PyObject* result = fn(a);
    if (!result) {
      throw PythonError();
    }
    return result;
  }
}

////////////////////////////////
// NodeBase
///////////////////////////////

struct NodeBase {
  PyObject_HEAD
  bool _erased;
  NodeBase* _prev;
  NodeBase* _next;
  PyObject* graph;
  PyObject* name;
  PyObject* op;
  PyObject* target;
  PyObject* type;
  PyObject* _input_nodes;
  PyObject* _args;
  PyObject* _kwargs;
  PyObject* users;
  PyObject* _repr_fn;
  PyObject* meta;
  PyObject* _sort_key;
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
  PyObject* graph = nullptr;
  PyObject* name = nullptr;
  PyObject* op = nullptr;
  PyObject* target = nullptr;
  PyObject* type = nullptr;
  if (!PyArg_ParseTuple(args, "OOOOO", &graph, &name, &op, &target, &type)) {
    return -1;
  }
  self->_erased = false;
  Py_INCREF(self);
  self->_prev = self;
  Py_INCREF(self);
  self->_next = self;
  self->graph = Py_NewRef(graph);
  self->name = Py_NewRef(name);
  self->op = Py_NewRef(op);
  self->target = Py_NewRef(target);
  self->type = Py_NewRef(type);
  self->_input_nodes = PyDict_New();
  self->_args = nullptr; // set with _update_args_kwargs
  self->_kwargs = nullptr; // set with _update_args_kwargs
  self->users = PyDict_New();
  self->_repr_fn = Py_NewRef(Py_None);
  self->meta = PyDict_New();
  self->_sort_key = PyTuple_New(0);
  return 0;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static struct PyMemberDef NodeBase_members[] = {
    {"_erased", T_BOOL, offsetof(NodeBase, _erased), 0, nullptr},
    {"_prev", T_OBJECT_EX, offsetof(NodeBase, _prev), 0, nullptr},
    {"_next", T_OBJECT_EX, offsetof(NodeBase, _next), 0, nullptr},
    {"graph", T_OBJECT_EX, offsetof(NodeBase, graph), 0, nullptr},
    {"name", T_OBJECT_EX, offsetof(NodeBase, name), 0, nullptr},
    {"op", T_OBJECT_EX, offsetof(NodeBase, op), 0, nullptr},
    {"target", T_OBJECT_EX, offsetof(NodeBase, target), 0, nullptr},
    {"type", T_OBJECT_EX, offsetof(NodeBase, type), 0, nullptr},
    {"_input_nodes", T_OBJECT_EX, offsetof(NodeBase, _input_nodes), 0, nullptr},
    {"_args", T_OBJECT_EX, offsetof(NodeBase, _args), 0, nullptr},
    {"_kwargs", T_OBJECT_EX, offsetof(NodeBase, _kwargs), 0, nullptr},
    {"users", T_OBJECT_EX, offsetof(NodeBase, users), 0, nullptr},
    {"_repr_fn", T_OBJECT_EX, offsetof(NodeBase, _repr_fn), 0, nullptr},
    {"meta", T_OBJECT_EX, offsetof(NodeBase, meta), 0, nullptr},
    {"_sort_key", T_OBJECT_EX, offsetof(NodeBase, _sort_key), 0, nullptr},
    {nullptr} /* Sentinel */
};

static int NodeBase_traverse(NodeBase* self, visitproc visit, void* arg) {
  Py_VISIT(self->_prev);
  Py_VISIT(self->_next);
  Py_VISIT(self->graph);
  Py_VISIT(self->name);
  Py_VISIT(self->op);
  Py_VISIT(self->target);
  Py_VISIT(self->type);
  Py_VISIT(self->_input_nodes);
  Py_VISIT(self->_args);
  Py_VISIT(self->_kwargs);
  Py_VISIT(self->users);
  Py_VISIT(self->_repr_fn);
  Py_VISIT(self->meta);
  Py_VISIT(self->_sort_key);
  return 0;
}

static int NodeBase_clear(NodeBase* self) {
  Py_CLEAR(self->_prev);
  Py_CLEAR(self->_next);
  Py_CLEAR(self->graph);
  Py_CLEAR(self->name);
  Py_CLEAR(self->op);
  Py_CLEAR(self->target);
  Py_CLEAR(self->type);
  Py_CLEAR(self->_input_nodes);
  Py_CLEAR(self->_args);
  Py_CLEAR(self->_kwargs);
  Py_CLEAR(self->users);
  Py_CLEAR(self->_repr_fn);
  Py_CLEAR(self->meta);
  Py_CLEAR(self->_sort_key);
  return 0;
}

static void NodeBase_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  (void)NodeBase_clear((NodeBase*)self);
  Py_TYPE(self)->tp_free(self);
}

static PyObject* NodeBase__update_args_kwargs(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  // Verify argument count
  if (nargs != 2) {
    PyErr_SetString(
        PyExc_TypeError,
        "_update_args_kwargs() requires exactly 2 arguments (new_args, new_kwargs)");
    return nullptr;
  }
  auto node = reinterpret_cast<NodeBase*>(self);
  auto input_nodes = node->_input_nodes;
  if (PyDict_GET_SIZE(input_nodes) > 0) {
    // Clear other.users containing us and input_nodes
    PyObject *key = nullptr, *value = nullptr; // borrowed
    Py_ssize_t pos = 0;
    while (PyDict_Next(input_nodes, &pos, &key, &value)) {
      // key.users.pop(self), intentionally ignore KeyError
      PyDict_DelItem(reinterpret_cast<NodeBase*>(key)->users, self);
    }
    PyDict_Clear(input_nodes);
  }

  auto visit_fn = [self, input_nodes](PyObject* x) {
    if (is_node(x)) {
      // self._input_nodes.setdefault(x)
      if (!PyDict_SetDefault(input_nodes, x, Py_None)) {
        throw PythonError();
      }
      // x.users.setdefault(self)
      if (!PyDict_SetDefault(
              reinterpret_cast<NodeBase*>(x)->users, self, Py_None)) {
        throw PythonError();
      }
    }
    return Py_NewRef(x);
  };

  // We do three things in a single pass of the args
  // - Normalize list->immutable_list, dict->immutable_dict, etc
  // - Populate self._input_nodes
  // - Populate arg.users[self] for each arg
  try {
    Py_CLEAR(node->_args);
    node->_args = map_aggregate(args[0], visit_fn);
    Py_CLEAR(node->_kwargs);
    node->_kwargs = map_aggregate(args[1], visit_fn);
    Py_RETURN_NONE;
  } catch (const PythonError& e) {
    return nullptr;
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef NodeBase_methods[] = {
    {"_update_args_kwargs",
     (PyCFunction)(void*)(NodeBase__update_args_kwargs),
     METH_FASTCALL,
     "Internal method: do not call directly."},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

PyTypeObject NodeBaseType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._NodeBase", /* tp_name */
    sizeof(NodeBase), /* tp_basicsize */
    0, /* tp_itemsize */
    NodeBase_dealloc, /* tp_dealloc */
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
    NodeBase_methods, /* tp_methods */
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

} // namespace

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
static PyObject* NodeIter_iternext_helper(NodeIter* self) {
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

static PyObject* NodeIter_iternext(PyObject* _self) {
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

////////////////////////////////
// Global methods
////////////////////////////////

static PyObject* py_map_aggregate(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(
        PyExc_TypeError, "map_aggregate() takes exactly two arguments");
    return nullptr;
  }
  try {
    PyObject* fn = args[1];
    // args[0]: aggregate, args[1]: callable fn
    return map_aggregate(
        args[0], [fn](PyObject* a) { return PyObject_CallOneArg(fn, a); });
  } catch (const PythonError& e) {
    return nullptr; // error should already be set
  }
}

static PyObject* py_map_arg(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(PyExc_TypeError, "map_arg() takes exactly two arguments");
    return nullptr;
  }
  try {
    PyObject* fn = args[1];
    // args[0]: aggregate, args[1]: callable fn
    return map_aggregate(args[0], [fn](PyObject* a) {
      if (is_node(a)) {
        return PyObject_CallOneArg(fn, a);
      }
      return Py_NewRef(a);
    });
  } catch (const PythonError& e) {
    return nullptr; // error should already be set
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef extra_methods[] = {
    {"_fx_map_aggregate",
     (PyCFunction)(void*)(py_map_aggregate),
     METH_FASTCALL,
     "Recursively apply a function to every element in an aggregate object."},
    {"_fx_map_arg",
     (PyCFunction)(void*)(py_map_arg),
     METH_FASTCALL,
     "Recursively apply a function to every Node in an aggregate object."},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

bool NodeBase_init(PyObject* module) {
  if (PyModule_AddType(module, &NodeBaseType) < 0) {
    return false;
  }
  if (PyModule_AddFunctions(module, extra_methods) < 0) {
    return false;
  }
  return true;
}

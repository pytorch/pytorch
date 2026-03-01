#include <torch/csrc/fx/node.h>

#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <structmember.h>
#include <torch/csrc/fx/graph.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <algorithm>
#include <sstream>
#include <string>

namespace {

using NodeSortKey = c10::SmallVector<int64_t, 4>;
struct NodeBase;

// Thrown to exit out of a C++ function and return an error to Python.
class PythonError : public std::exception {};

// Legal ops for validation
static const char* legal_ops[] = {
    "placeholder",
    "call_method",
    "call_module",
    "call_function",
    "get_attr",
    "output",
    "root",
    nullptr};

inline static bool is_legal_op(PyObject* op) {
  if (!PyUnicode_Check(op)) {
    return false;
  }
  Py_ssize_t size = 0;
  const char* op_str = PyUnicode_AsUTF8AndSize(op, &size);
  if (!op_str) {
    PyErr_Clear();
    return false;
  }
  for (const char** p = legal_ops; *p != nullptr; ++p) {
    if (strcmp(op_str, *p) == 0) {
      return true;
    }
  }
  return false;
}

// Set a ValueError with torch.typename for target type mismatches in Node init.
// Returns -1 (the init failure code) for convenience.
inline static int set_target_type_error(
    PyObject* graph,
    PyObject* name,
    PyObject* target,
    const char* expected) {
  THPObjectPtr torch_module(PyImport_ImportModule("torch"));
  if (!torch_module) {
    return -1;
  }
  THPObjectPtr typename_fn(PyObject_GetAttrString(torch_module, "typename"));
  if (!typename_fn) {
    return -1;
  }
  THPObjectPtr type_str(PyObject_CallOneArg(typename_fn, target));
  const char* type_cstr = type_str ? PyUnicode_AsUTF8(type_str) : "unknown";
  PyErr_Format(
      PyExc_ValueError,
      "Node [graph = %R, name = '%S'] target %R has type %s "
      "but a %s is expected",
      graph,
      name,
      target,
      type_cstr,
      expected);
  return -1;
}

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

// Forward declarations
struct NodeBase;
static PyObject* NodeBase__update_args_kwargs(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs);

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
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  alignas(NodeSortKey) char sort_key_buf[sizeof(NodeSortKey)];

  inline NodeSortKey& sort_key() {
    return *reinterpret_cast<NodeSortKey*>(sort_key_buf);
  }

  inline void set_prev(NodeBase* value) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(value);
    Py_SETREF(_prev, reinterpret_cast<NodeBase*>(Py_NewRef(value)));
  }

  inline void set_next(NodeBase* value) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(value);
    Py_SETREF(_next, reinterpret_cast<NodeBase*>(Py_NewRef(value)));
  }

  // Equivalent to:
  //   p, n = self._prev, self._next
  //   p._next, n._prev = n, p
  inline void remove_from_list() {
    if (this->_prev == this && this->_next == this) {
      return;
    }
    NodeBase* p = this->_prev;
    NodeBase* n = this->_next;
    p->set_next(n);
    n->set_prev(p);
  }
};

static PyObject* NodeBase_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  PyObject* self = type->tp_alloc(type, 0);
  if (!self)
    return nullptr;
  new (reinterpret_cast<NodeBase*>(self)->sort_key_buf)
      NodeSortKey(); // placement new does not allocate
  return self;
}

static int NodeBase_init_fn(NodeBase* self, PyObject* args, PyObject* kwds) {
  PyObject* graph = nullptr;
  PyObject* name = nullptr;
  PyObject* op = nullptr;
  PyObject* target = nullptr;
  PyObject* type = Py_None; // default to None
  PyObject* init_args = nullptr;
  PyObject* init_kwargs = nullptr;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* keywords[] = {
      "graph",
      "name",
      "op",
      "target",
      "args",
      "kwargs",
      "return_type",
      nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwds,
          "OOOOOO|O",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(keywords),
          &graph,
          &name,
          &op,
          &target,
          &init_args,
          &init_kwargs,
          &type)) {
    return -1;
  }

  // Validate op and target
  Py_ssize_t op_size = 0;
  const char* op_str = PyUnicode_AsUTF8AndSize(op, &op_size);
  if (!op_str) {
    return -1;
  }

  if (strcmp(op_str, "call_function") == 0) {
    if (!PyCallable_Check(target)) {
      return set_target_type_error(graph, name, target, "Callable");
    }
  } else {
    if (!is_legal_op(op)) {
      PyErr_Format(PyExc_AssertionError, "Invalid op: %S", op);
      return -1;
    }
    if (!PyUnicode_Check(target)) {
      return set_target_type_error(graph, name, target, "str");
    }
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

  // Call _update_args_kwargs with the provided args and kwargs
  PyObject* update_args[2] = {init_args, init_kwargs};
  THPObjectPtr result(
      NodeBase__update_args_kwargs((PyObject*)self, update_args, 2));
  if (!result) {
    return -1;
  }

  return 0;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static struct PyMemberDef NodeBase_members[] = {
    {"_erased", T_BOOL, offsetof(NodeBase, _erased), 0, nullptr},
    {"_prev", T_OBJECT, offsetof(NodeBase, _prev), 0, nullptr},
    {"_next", T_OBJECT, offsetof(NodeBase, _next), 0, nullptr},
    {"graph", T_OBJECT, offsetof(NodeBase, graph), 0, nullptr},
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
  return 0;
}

static void NodeBase_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  reinterpret_cast<NodeBase*>(self)->sort_key().~NodeSortKey();
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
  //
  // Note: We must create the new args and kwargs before clearing the old ones,
  // because args[0] or args[1] might alias node->_args or node->_kwargs
  // (e.g., when only setting args, we pass node->_kwargs as args[1]).
  try {
    PyObject* new_args = map_aggregate(args[0], visit_fn);
    PyObject* new_kwargs = map_aggregate(args[1], visit_fn);
    Py_CLEAR(node->_args);
    Py_CLEAR(node->_kwargs);
    node->_args = new_args;
    node->_kwargs = new_kwargs;
    Py_RETURN_NONE;
  } catch (const PythonError&) {
    return nullptr;
  }
}

static PyObject* NodeBase__remove_from_list(
    PyObject* self,
    PyObject* _ignored) {
  reinterpret_cast<NodeBase*>(self)->remove_from_list();
  Py_RETURN_NONE;
}

static PyObject* NodeBase__replace_input_with(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(
        PyExc_TypeError,
        "_replace_input_with() requires exactly 2 arguments (old_input, new_input)");
    return nullptr;
  }
  PyObject* old_input = args[0];
  PyObject* new_input = args[1];
  auto replace_fn = [old_input, new_input](PyObject* maybe_node) {
    if (maybe_node == old_input) {
      return Py_NewRef(new_input);
    }
    return Py_NewRef(maybe_node);
  };

  auto node = reinterpret_cast<NodeBase*>(self);
  try {
    THPObjectPtr new_args(map_aggregate(node->_args, replace_fn));
    if (!new_args) {
      return nullptr;
    }
    THPObjectPtr new_kwargs(map_aggregate(node->_kwargs, replace_fn));
    if (!new_kwargs) {
      return nullptr;
    }

    PyObject* update_args[2] = {new_args.get(), new_kwargs.get()};
    return NodeBase__update_args_kwargs(self, update_args, 2);
  } catch (const PythonError&) {
    return nullptr;
  }
}

static PyObject* NodeBase__prepend(PyObject* self_, PyObject* arg) {
  if (self_ == arg) {
    Py_RETURN_NONE;
  }
  if (!is_node(arg)) {
    PyErr_SetString(PyExc_TypeError, "_prepend() argument must be a Node");
    return nullptr;
  }
  NodeBase* self = reinterpret_cast<NodeBase*>(self_);
  NodeBase* x = reinterpret_cast<NodeBase*>(arg);
  if (self->graph != x->graph) {
    PyErr_SetString(
        PyExc_AssertionError,
        "Attempting to move a Node into a different Graph");
    return nullptr;
  }

  x->remove_from_list();
  NodeBase* p = self->_prev;
  p->set_next(x);
  x->set_prev(p);
  x->set_next(self);
  self->set_prev(x);

  // Now compute x.sort_key()
  const NodeSortKey& psk = x->_prev->sort_key();
  const NodeSortKey& nsk = x->_next->sort_key();
  if (psk.size() > nsk.size()) {
    // prefix = psk[: len(nsk)+1]
    size_t slice_len = nsk.size() + 1;
    NodeSortKey prefix(psk.begin(), psk.begin() + slice_len);
    // last element is idx => increment by 1
    prefix.back()++;
    x->sort_key() = std::move(prefix);
  } else if (psk.size() < nsk.size()) {
    // prefix = nsk[: len(psk)+1]
    size_t slice_len = psk.size() + 1;
    NodeSortKey prefix(nsk.begin(), nsk.begin() + slice_len);
    // last element is idx => decrement by 1
    prefix.back()--;
    x->sort_key() = std::move(prefix);
  } else {
    // same length => add a 0
    x->sort_key() = psk;
    x->sort_key().emplace_back(0);
  }
  Py_RETURN_NONE;
}

// __lt__(self, other): Return self.sort_key < other.sort_key
static PyObject* NodeBase___lt__(PyObject* self, PyObject* other) {
  // METH_O => one argument: 'other'
  if (!is_node(other)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  const NodeSortKey& lhs = reinterpret_cast<NodeBase*>(self)->sort_key();
  const NodeSortKey& rhs = reinterpret_cast<NodeBase*>(other)->sort_key();
  bool less = std::lexicographical_compare(
      lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
  if (less)
    Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

// __gt__(self, other): Return self.sort_key() > other.sort_key
static PyObject* NodeBase___gt__(PyObject* self, PyObject* other) {
  if (!is_node(other)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  const NodeSortKey& lhs = reinterpret_cast<NodeBase*>(self)->sort_key();
  const NodeSortKey& rhs = reinterpret_cast<NodeBase*>(other)->sort_key();
  // "a > b" is equivalent to "b < a"
  bool greater = std::lexicographical_compare(
      rhs.begin(), rhs.end(), lhs.begin(), lhs.end());
  if (greater)
    Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

static PyObject* NodeBase___ge__(PyObject* self, PyObject* other) {
  if (self == other) {
    Py_RETURN_TRUE;
  }
  return NodeBase___gt__(self, other);
}

// __le__(self, other): Return not (self > other)
static PyObject* NodeBase___le__(PyObject* self, PyObject* other) {
  if (self == other) {
    Py_RETURN_TRUE;
  }
  return NodeBase___lt__(self, other);
}

// prepend(x): Insert x before this node in the list
static PyObject* NodeBase_prepend(PyObject* self, PyObject* arg) {
  return NodeBase__prepend(self, arg);
}

// append(x): Insert x after this node in the list
static PyObject* NodeBase_append(PyObject* self, PyObject* arg) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  return NodeBase__prepend(reinterpret_cast<PyObject*>(node->_next), arg);
}

// update_arg(idx, arg): Update existing positional argument at idx
static PyObject* NodeBase_update_arg(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(
        PyExc_TypeError,
        "update_arg() requires exactly 2 arguments (idx, arg)");
    return nullptr;
  }
  NodeBase* node = reinterpret_cast<NodeBase*>(self);

  Py_ssize_t idx = PyLong_AsSsize_t(args[0]);
  if (idx == -1 && PyErr_Occurred()) {
    return nullptr;
  }
  PyObject* arg = args[1];

  // args = list(self.args); args[idx] = arg; self.args = tuple(args)
  Py_ssize_t args_len = PyTuple_GET_SIZE(node->_args);
  if (idx < 0 || idx >= args_len) {
    PyErr_SetString(PyExc_IndexError, "update_arg index out of range");
    return nullptr;
  }

  THPObjectPtr new_args(PyList_New(args_len));
  if (!new_args) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < args_len; i++) {
    PyObject* item = (i == idx) ? arg : PyTuple_GET_ITEM(node->_args, i);
    Py_INCREF(item);
    PyList_SET_ITEM(new_args.get(), i, item);
  }
  THPObjectPtr new_args_tuple(PyList_AsTuple(new_args.get()));
  if (!new_args_tuple) {
    return nullptr;
  }

  PyObject* update_args_arr[2] = {new_args_tuple.get(), node->_kwargs};
  return NodeBase__update_args_kwargs(self, update_args_arr, 2);
}

// insert_arg(idx, arg): Insert positional argument at idx
static PyObject* NodeBase_insert_arg(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(
        PyExc_TypeError,
        "insert_arg() requires exactly 2 arguments (idx, arg)");
    return nullptr;
  }
  NodeBase* node = reinterpret_cast<NodeBase*>(self);

  Py_ssize_t idx = PyLong_AsSsize_t(args[0]);
  if (idx == -1 && PyErr_Occurred()) {
    return nullptr;
  }
  PyObject* arg = args[1];

  Py_ssize_t args_len = PyTuple_GET_SIZE(node->_args);
  if (idx < 0 || idx > args_len) {
    PyErr_SetString(
        PyExc_AssertionError,
        "insert_args index must be between 0 and len(self.args)");
    return nullptr;
  }

  // Build new args tuple: args_left + (arg,) + args_right
  THPObjectPtr new_args(PyTuple_New(args_len + 1));
  if (!new_args) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < idx; i++) {
    PyObject* item = PyTuple_GET_ITEM(node->_args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(new_args.get(), i, item);
  }
  Py_INCREF(arg);
  PyTuple_SET_ITEM(new_args.get(), idx, arg);
  for (Py_ssize_t i = idx; i < args_len; i++) {
    PyObject* item = PyTuple_GET_ITEM(node->_args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(new_args.get(), i + 1, item);
  }

  // Update _args directly and track new input nodes
  Py_SETREF(node->_args, new_args.release());

  // Track new input nodes from the inserted arg
  THPObjectPtr new_input_nodes(PyDict_New());
  if (!new_input_nodes) {
    return nullptr;
  }
  try {
    auto track_fn = [&new_input_nodes](PyObject* x) {
      if (is_node(x)) {
        if (!PyDict_SetDefault(new_input_nodes.get(), x, Py_None)) {
          throw PythonError();
        }
      }
      return Py_NewRef(x);
    };
    THPObjectPtr mapped(map_aggregate(arg, track_fn));
  } catch (const PythonError&) {
    return nullptr;
  }

  // Update _input_nodes and users for new nodes
  PyObject *key = nullptr, *value = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(new_input_nodes.get(), &pos, &key, &value)) {
    // Check if not already in _input_nodes
    if (!PyDict_Contains(node->_input_nodes, key)) {
      if (PyDict_SetDefault(node->_input_nodes, key, Py_None) == nullptr) {
        return nullptr;
      }
      if (PyDict_SetDefault(
              reinterpret_cast<NodeBase*>(key)->users, self, Py_None) ==
          nullptr) {
        return nullptr;
      }
    }
  }

  Py_RETURN_NONE;
}

// update_kwarg(key, arg): Update existing keyword argument
static PyObject* NodeBase_update_kwarg(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(
        PyExc_TypeError,
        "update_kwarg() requires exactly 2 arguments (key, arg)");
    return nullptr;
  }
  NodeBase* node = reinterpret_cast<NodeBase*>(self);

  PyObject* key = args[0];
  PyObject* arg = args[1];

  // self.kwargs = {**self.kwargs, key: arg}
  THPObjectPtr new_kwargs(PyDict_Copy(node->_kwargs));
  if (!new_kwargs) {
    return nullptr;
  }
  if (PyDict_SetItem(new_kwargs.get(), key, arg) < 0) {
    return nullptr;
  }

  PyObject* update_args_arr[2] = {node->_args, new_kwargs.get()};
  return NodeBase__update_args_kwargs(self, update_args_arr, 2);
}

// replace_input_with(old_input, new_input): Loop through input nodes and
// replace
static bool call_replace_hook_with_keywords(
    PyObject* hook,
    PyObject* old_node,
    PyObject* new_name,
    PyObject* user) {
  THPObjectPtr kwargs(PyDict_New());
  if (!kwargs) {
    return false;
  }
  if (PyDict_SetItemString(kwargs.get(), "old", old_node) < 0 ||
      PyDict_SetItemString(kwargs.get(), "new", new_name) < 0 ||
      PyDict_SetItemString(kwargs.get(), "user", user) < 0) {
    return false;
  }
  THPObjectPtr args(PyTuple_New(0));
  if (!args) {
    return false;
  }
  THPObjectPtr hook_result(PyObject_Call(hook, args.get(), kwargs.get()));
  if (!hook_result) {
    return false;
  }
  return true;
}

static PyObject* NodeBase_replace_input_with(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(
        PyExc_TypeError,
        "replace_input_with() requires exactly 2 arguments (old_input, new_input)");
    return nullptr;
  }
  NodeBase* node = reinterpret_cast<NodeBase*>(self);

  PyObject* old_input = args[0];
  PyObject* new_input = args[1];

  // Check for replace hooks on the graph's owning module
  // Note: Only access .name if there are actually hooks to call, matching
  // Python's behavior where `if replace_hooks:` is falsy for empty list.
  PyObject* owning_module = GraphBase_borrow_owning_module(node->graph);
  if (owning_module && owning_module != Py_None) {
    THPObjectPtr replace_hooks(
        PyObject_GetAttrString(owning_module, "_replace_hooks"));
    if (replace_hooks && replace_hooks.get() != Py_None &&
        PySequence_Check(replace_hooks)) {
      Py_ssize_t num_hooks = PySequence_Size(replace_hooks);
      if (num_hooks > 0) {
        THPObjectPtr new_input_name(PyObject_GetAttrString(new_input, "name"));
        if (!new_input_name) {
          return nullptr;
        }
        for (Py_ssize_t i = 0; i < num_hooks; i++) {
          THPObjectPtr hook(PySequence_GetItem(replace_hooks, i));
          if (hook &&
              !call_replace_hook_with_keywords(
                  hook.get(), old_input, new_input_name.get(), self)) {
            return nullptr;
          }
        }
      }
    }
    PyErr_Clear(); // Clear any errors from attribute access
  }

  return NodeBase__replace_input_with(self, args, nargs);
}

// Convert the NodeBase::sort_key vector<long> into a Python tuple of ints
// Only used by pickle/__getstate__
static PyObject* NodeBase_get_sort_key(PyObject* self, void* /*closure*/) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  const NodeSortKey& vec = node->sort_key();
  Py_ssize_t n = static_cast<Py_ssize_t>(vec.size());
  THPObjectPtr tuple(PyTuple_New(n));
  if (!tuple) {
    return nullptr; // Out of memory
  }
  for (Py_ssize_t i = 0; i < n; i++) {
    PyObject* value = PyLong_FromSsize_t(vec[i]);
    if (!value) {
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple.get(), i, value);
  }
  return tuple.release();
}

// Setter for NodeBase::sort_key: expects a Python tuple of ints, e.g.
// node._sort_key = (1,2,3) Only used by pickle/__setstate__
static int NodeBase_set_sort_key(
    PyObject* self,
    PyObject* value,
    void* /*closure*/) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  if (!PyTuple_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "_sort_key must be an tuple of ints");
    return -1;
  }
  Py_ssize_t size = PyTuple_GET_SIZE(value);
  NodeSortKey new_vec;
  new_vec.reserve(size);
  for (Py_ssize_t i = 0; i < size; i++) {
    int64_t val = PyLong_AsSsize_t(PyTuple_GET_ITEM(value, i));
    if (val == -1 && PyErr_Occurred()) {
      return -1;
    }
    new_vec.emplace_back(val);
  }
  node->sort_key() = std::move(new_vec);
  return 0;
}

// replace_all_uses_with(replace_with, delete_user_cb=None, *,
// propagate_meta=False)
static PyObject* NodeBase_replace_all_uses_with(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);

  // Parse arguments
  PyObject* replace_with = nullptr;
  PyObject* delete_user_cb = Py_None;
  int propagate_meta = 0;

  // Handle positional arguments
  if (nargs >= 1) {
    replace_with = args[0];
  }
  if (nargs >= 2) {
    delete_user_cb = args[1];
  }

  // Parse keyword arguments
  if (kwnames) {
    Py_ssize_t nkwargs = PyTuple_GET_SIZE(kwnames);
    for (Py_ssize_t i = 0; i < nkwargs; i++) {
      PyObject* key = PyTuple_GET_ITEM(kwnames, i);
      const char* key_str = PyUnicode_AsUTF8(key);
      if (!key_str) {
        return nullptr;
      }
      PyObject* kwval = args[nargs + i];
      if (strcmp(key_str, "replace_with") == 0) {
        if (replace_with) {
          PyErr_SetString(
              PyExc_TypeError,
              "replace_all_uses_with() got multiple values for argument 'replace_with'");
          return nullptr;
        }
        replace_with = kwval;
      } else if (strcmp(key_str, "delete_user_cb") == 0) {
        if (nargs >= 2) {
          PyErr_SetString(
              PyExc_TypeError,
              "replace_all_uses_with() got multiple values for argument 'delete_user_cb'");
          return nullptr;
        }
        delete_user_cb = kwval;
      } else if (strcmp(key_str, "propagate_meta") == 0) {
        propagate_meta = PyObject_IsTrue(kwval);
        if (propagate_meta == -1) {
          return nullptr;
        }
      } else {
        PyErr_Format(
            PyExc_TypeError,
            "replace_all_uses_with() got an unexpected keyword argument '%s'",
            key_str);
        return nullptr;
      }
    }
  }

  // Check required argument
  if (!replace_with) {
    PyErr_SetString(
        PyExc_TypeError,
        "replace_all_uses_with() missing required argument: 'replace_with'");
    return nullptr;
  }

  // propagate_meta copies self.meta to replace_with.meta
  // Access .meta via attribute lookup to match Python semantics (raises
  // AttributeError if replace_with doesn't have .meta)
  if (propagate_meta) {
    THPObjectPtr replace_with_meta(
        PyObject_GetAttrString(replace_with, "meta"));
    if (!replace_with_meta) {
      return nullptr; // Raises AttributeError naturally
    }
    if (PyDict_Size(replace_with_meta.get()) != 0) {
      PyErr_SetString(
          PyExc_AssertionError,
          "Called node.replace_all_uses_with(replace_with, propagate_meta=True), "
          "but replace_with already has .meta keys");
      return nullptr;
    }
    PyObject *key = nullptr, *value = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(node->meta, &pos, &key, &value)) {
      if (PyDict_SetItem(replace_with_meta.get(), key, value) < 0) {
        return nullptr;
      }
    }
  }

  // Get list of users to process
  THPObjectPtr to_process(PyDict_Keys(node->users));
  if (!to_process) {
    return nullptr;
  }

  // Get replace hooks
  THPObjectPtr replace_hooks;
  PyObject* owning_module = GraphBase_borrow_owning_module(node->graph);
  if (owning_module && owning_module != Py_None) {
    replace_hooks =
        THPObjectPtr(PyObject_GetAttrString(owning_module, "_replace_hooks"));
    if (!replace_hooks || replace_hooks.get() == Py_None) {
      replace_hooks = THPObjectPtr();
    }
    PyErr_Clear();
  }
  PyErr_Clear();

  // Result list
  THPObjectPtr result(PyList_New(0));
  if (!result) {
    return nullptr;
  }

  // Fetch replace_with.name once before the loop if there are hooks to call.
  THPObjectPtr replace_with_name;
  if (replace_hooks && PySequence_Check(replace_hooks.get()) &&
      PySequence_Size(replace_hooks.get()) > 0) {
    replace_with_name =
        THPObjectPtr(PyObject_GetAttrString(replace_with, "name"));
    if (!replace_with_name) {
      return nullptr;
    }
  }

  Py_ssize_t num_users = PyList_GET_SIZE(to_process.get());
  for (Py_ssize_t i = 0; i < num_users; i++) {
    PyObject* use_node = PyList_GET_ITEM(to_process.get(), i);

    // Check delete_user_cb
    if (delete_user_cb != Py_None) {
      THPObjectPtr cb_result(PyObject_CallOneArg(delete_user_cb, use_node));
      if (!cb_result) {
        return nullptr;
      }
      int should_continue = PyObject_IsTrue(cb_result);
      if (should_continue == -1) {
        return nullptr;
      }
      if (!should_continue) {
        continue;
      }
    }

    // Add to result
    if (PyList_Append(result.get(), use_node) < 0) {
      return nullptr;
    }

    // Call replace hooks
    if (replace_with_name) {
      Py_ssize_t num_hooks = PySequence_Size(replace_hooks.get());
      for (Py_ssize_t j = 0; j < num_hooks; j++) {
        THPObjectPtr hook(PySequence_GetItem(replace_hooks.get(), j));
        if (hook &&
            !call_replace_hook_with_keywords(
                hook.get(), self, replace_with_name.get(), use_node)) {
          return nullptr;
        }
      }
    }

    // Call _replace_input_with on use_node
    PyObject* replace_args[2] = {self, replace_with};
    THPObjectPtr replace_result(
        NodeBase__replace_input_with(use_node, replace_args, 2));
    if (!replace_result) {
      return nullptr;
    }
  }

  return result.release();
}

// is_impure(impure_random=True): Check if this op is impure
static PyObject* NodeBase_is_impure(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);

  int impure_random = 1; // default True
  if (nargs >= 1) {
    impure_random = PyObject_IsTrue(args[0]);
    if (impure_random == -1) {
      return nullptr;
    }
  }
  if (kwnames) {
    Py_ssize_t nkwargs = PyTuple_GET_SIZE(kwnames);
    for (Py_ssize_t i = 0; i < nkwargs; i++) {
      PyObject* key = PyTuple_GET_ITEM(kwnames, i);
      const char* key_str = PyUnicode_AsUTF8(key);
      if (key_str && strcmp(key_str, "impure_random") == 0) {
        impure_random = PyObject_IsTrue(args[nargs + i]);
        if (impure_random == -1) {
          return nullptr;
        }
      }
    }
  }

  const char* op_str = PyUnicode_AsUTF8(node->op);
  if (!op_str) {
    return nullptr;
  }

  // Placeholders and outputs are always impure
  if (strcmp(op_str, "placeholder") == 0 || strcmp(op_str, "output") == 0) {
    Py_RETURN_TRUE;
  }

  // Check if impure module
  if (strcmp(op_str, "call_module") == 0) {
    PyObject* owning_module = GraphBase_borrow_owning_module(node->graph);
    if (!owning_module || owning_module == Py_None) {
      PyErr_SetString(
          PyExc_AssertionError,
          "self.graph.owning_module not set for purity check");
      return nullptr;
    }
    THPObjectPtr get_submodule(
        PyObject_GetAttrString(owning_module, "get_submodule"));
    if (!get_submodule) {
      return nullptr;
    }
    THPObjectPtr target_mod(
        PyObject_CallOneArg(get_submodule.get(), node->target));
    if (!target_mod) {
      return nullptr;
    }
    if (target_mod.get() == Py_None) {
      PyErr_Format(
          PyExc_AssertionError,
          "Did not find expected submodule target %S",
          node->target);
      return nullptr;
    }
    THPObjectPtr is_impure_attr(
        PyObject_GetAttrString(target_mod.get(), "_is_impure"));
    if (is_impure_attr) {
      int is_impure_val = PyObject_IsTrue(is_impure_attr);
      if (is_impure_val == -1) {
        PyErr_Clear();
        Py_RETURN_FALSE;
      }
      if (is_impure_val) {
        Py_RETURN_TRUE;
      }
    }
    PyErr_Clear();
    Py_RETURN_FALSE;
  }

  // For call_function, delegate to torch._library.utils.is_impure
  if (strcmp(op_str, "call_function") == 0) {
    THPObjectPtr utils_module(PyImport_ImportModule("torch._library.utils"));
    if (!utils_module) {
      return nullptr;
    }
    THPObjectPtr is_impure_fn(
        PyObject_GetAttrString(utils_module.get(), "is_impure"));
    if (!is_impure_fn) {
      return nullptr;
    }

    // Build keyword arguments dict
    THPObjectPtr kwargs_dict(PyDict_New());
    if (!kwargs_dict) {
      return nullptr;
    }
    if (PyDict_SetItemString(kwargs_dict.get(), "args", node->_args) < 0) {
      return nullptr;
    }
    if (PyDict_SetItemString(kwargs_dict.get(), "kwargs", node->_kwargs) < 0) {
      return nullptr;
    }
    PyObject* impure_random_obj = impure_random ? Py_True : Py_False;
    if (PyDict_SetItemString(
            kwargs_dict.get(), "impure_random", impure_random_obj) < 0) {
      return nullptr;
    }

    // Build positional args tuple with just the target
    THPObjectPtr args_tuple(PyTuple_New(1));
    if (!args_tuple) {
      return nullptr;
    }
    Py_INCREF(node->target);
    PyTuple_SET_ITEM(args_tuple.get(), 0, node->target);

    THPObjectPtr result(
        PyObject_Call(is_impure_fn.get(), args_tuple.get(), kwargs_dict.get()));
    if (!result) {
      return nullptr;
    }
    return result.release();
  }

  Py_RETURN_FALSE;
}

// _rename(candidate): Update name through graph namespace
static PyObject* NodeBase__rename(PyObject* self, PyObject* candidate) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);

  // Check if candidate equals current name
  int eq = PyObject_RichCompareBool(candidate, node->name, Py_EQ);
  if (eq == -1) {
    return nullptr;
  }
  if (eq) {
    Py_RETURN_NONE;
  }

  // Get graph._graph_namespace
  THPObjectPtr graph_namespace(
      PyObject_GetAttrString(node->graph, "_graph_namespace"));
  if (!graph_namespace) {
    return nullptr;
  }

  // Call create_name(candidate, None)
  THPObjectPtr create_name(
      PyObject_GetAttrString(graph_namespace.get(), "create_name"));
  if (!create_name) {
    return nullptr;
  }
  THPObjectPtr new_name(
      PyObject_CallFunction(create_name.get(), "OO", candidate, Py_None));
  if (!new_name) {
    return nullptr;
  }

  // Set self.name = new_name via __setattr__ so replace hooks fire
  if (PyObject_SetAttrString(self, "name", new_name.get()) < 0) {
    return nullptr;
  }

  // Call graph._graph_namespace._rename_object(self, name)
  THPObjectPtr rename_obj(
      PyObject_GetAttrString(graph_namespace.get(), "_rename_object"));
  if (!rename_obj) {
    return nullptr;
  }
  THPObjectPtr rename_result(
      PyObject_CallFunction(rename_obj.get(), "OO", self, node->name));
  if (!rename_result) {
    return nullptr;
  }

  Py_RETURN_NONE;
}

// __getstate__: Return dict for pickling
static PyObject* NodeBase_getstate(
    PyObject* self,
    PyObject* Py_UNUSED(ignored)) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);

  THPObjectPtr dict(PyDict_New());
  if (!dict) {
    return nullptr;
  }

  // Get instance __dict__ if any
  THPObjectPtr instance_dict(PyObject_GetAttrString(self, "__dict__"));
  if (instance_dict) {
    if (PyDict_Update(dict.get(), instance_dict.get()) < 0) {
      return nullptr;
    }
  }
  PyErr_Clear();

  // graph, _prev, _next are omitted — the owning Graph restores them
  // in Graph.__setstate__; standalone Node pickle produces a detached
  // node (with these fields as None via T_OBJECT NULL semantics).
  if (PyDict_SetItemString(dict.get(), "name", node->name) < 0)
    return nullptr;
  if (PyDict_SetItemString(dict.get(), "op", node->op) < 0)
    return nullptr;
  if (PyDict_SetItemString(dict.get(), "target", node->target) < 0)
    return nullptr;
  if (PyDict_SetItemString(dict.get(), "type", node->type) < 0)
    return nullptr;

  THPObjectPtr sort_key(NodeBase_get_sort_key(self, nullptr));
  if (!sort_key) {
    return nullptr;
  }
  if (PyDict_SetItemString(dict.get(), "_sort_key", sort_key.get()) < 0)
    return nullptr;

  if (PyDict_SetItemString(dict.get(), "_args", node->_args) < 0)
    return nullptr;
  if (PyDict_SetItemString(dict.get(), "_kwargs", node->_kwargs) < 0)
    return nullptr;

  THPObjectPtr erased(PyBool_FromLong(node->_erased));
  if (PyDict_SetItemString(dict.get(), "_erased", erased.get()) < 0)
    return nullptr;
  if (PyDict_SetItemString(dict.get(), "_input_nodes", node->_input_nodes) < 0)
    return nullptr;
  if (PyDict_SetItemString(dict.get(), "users", node->users) < 0)
    return nullptr;
  if (PyDict_SetItemString(dict.get(), "_repr_fn", node->_repr_fn) < 0)
    return nullptr;
  if (PyDict_SetItemString(dict.get(), "meta", node->meta) < 0)
    return nullptr;

  return dict.release();
}

// __setstate__: Restore from dict
static PyObject* NodeBase_setstate(PyObject* self, PyObject* state) {
  if (!PyDict_Check(state)) {
    PyErr_SetString(PyExc_TypeError, "state must be a dict");
    return nullptr;
  }

  PyObject *key = nullptr, *value = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(state, &pos, &key, &value)) {
    if (PyObject_SetAttr(self, key, value) < 0) {
      return nullptr;
    }
  }

  Py_RETURN_NONE;
}

// normalized_arguments: Calls Python helper functions
static PyObject* NodeBase_normalized_arguments(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);

  // Parse arguments
  PyObject* root = nullptr;
  PyObject* arg_types = Py_None;
  PyObject* kwarg_types = Py_None;
  int normalize_to_only_use_kwargs = 0;

  if (nargs < 1) {
    PyErr_SetString(
        PyExc_TypeError,
        "normalized_arguments() requires at least 1 argument (root)");
    return nullptr;
  }
  root = args[0];
  if (nargs >= 2)
    arg_types = args[1];
  if (nargs >= 3)
    kwarg_types = args[2];
  if (nargs >= 4) {
    normalize_to_only_use_kwargs = PyObject_IsTrue(args[3]);
    if (normalize_to_only_use_kwargs == -1) {
      return nullptr;
    }
  }

  // Parse keyword arguments
  if (kwnames) {
    Py_ssize_t nkwargs = PyTuple_GET_SIZE(kwnames);
    for (Py_ssize_t i = 0; i < nkwargs; i++) {
      PyObject* kw_key = PyTuple_GET_ITEM(kwnames, i);
      const char* key_str = PyUnicode_AsUTF8(kw_key);
      if (!key_str)
        continue;
      if (strcmp(key_str, "arg_types") == 0) {
        arg_types = args[nargs + i];
      } else if (strcmp(key_str, "kwarg_types") == 0) {
        kwarg_types = args[nargs + i];
      } else if (strcmp(key_str, "normalize_to_only_use_kwargs") == 0) {
        normalize_to_only_use_kwargs = PyObject_IsTrue(args[nargs + i]);
        if (normalize_to_only_use_kwargs == -1) {
          return nullptr;
        }
      }
    }
  }

  const char* op_str = PyUnicode_AsUTF8(node->op);
  if (!op_str) {
    return nullptr;
  }

  if (strcmp(op_str, "call_function") == 0) {
    // Import and call normalize_function
    THPObjectPtr schemas_module(
        PyImport_ImportModule("torch.fx.operator_schemas"));
    if (!schemas_module) {
      return nullptr;
    }
    THPObjectPtr normalize_fn(
        PyObject_GetAttrString(schemas_module.get(), "normalize_function"));
    if (!normalize_fn) {
      return nullptr;
    }
    THPObjectPtr normalize_kwarg(PyBool_FromLong(normalize_to_only_use_kwargs));
    return PyObject_CallFunction(
        normalize_fn.get(),
        "OOOOOO",
        node->target,
        node->_args,
        node->_kwargs,
        arg_types,
        kwarg_types,
        normalize_kwarg.get());
  } else if (strcmp(op_str, "call_module") == 0) {
    // Import and call normalize_module
    THPObjectPtr schemas_module(
        PyImport_ImportModule("torch.fx.operator_schemas"));
    if (!schemas_module) {
      return nullptr;
    }
    THPObjectPtr normalize_fn(
        PyObject_GetAttrString(schemas_module.get(), "normalize_module"));
    if (!normalize_fn) {
      return nullptr;
    }
    THPObjectPtr normalize_kwarg(PyBool_FromLong(normalize_to_only_use_kwargs));
    return PyObject_CallFunction(
        normalize_fn.get(),
        "OOOOO",
        root,
        node->target,
        node->_args,
        node->_kwargs,
        normalize_kwarg.get());
  }

  Py_RETURN_NONE;
}

// format_node: Complex formatting - delegate to Python helper
static PyObject* NodeBase_format_node(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames) {
  // This is complex enough that we'll call a Python helper
  THPObjectPtr node_module(PyImport_ImportModule("torch.fx.node"));
  if (!node_module) {
    return nullptr;
  }
  THPObjectPtr format_node_helper(
      PyObject_GetAttrString(node_module.get(), "_format_node_impl"));
  if (!format_node_helper) {
    // Fallback: return basic format
    NodeBase* node = reinterpret_cast<NodeBase*>(self);
    return PyUnicode_FromFormat("%%%s", PyUnicode_AsUTF8(node->name));
  }

  // Call the helper with all arguments
  THPObjectPtr call_args(PyTuple_New(nargs + 1));
  if (!call_args) {
    return nullptr;
  }
  Py_INCREF(self);
  PyTuple_SET_ITEM(call_args.get(), 0, self);
  for (Py_ssize_t i = 0; i < nargs; i++) {
    Py_INCREF(args[i]);
    PyTuple_SET_ITEM(call_args.get(), i + 1, args[i]);
  }

  THPObjectPtr kwargs_dict;
  if (kwnames) {
    kwargs_dict = THPObjectPtr(PyDict_New());
    if (!kwargs_dict) {
      return nullptr;
    }
    Py_ssize_t nkwargs = PyTuple_GET_SIZE(kwnames);
    for (Py_ssize_t i = 0; i < nkwargs; i++) {
      PyObject* key = PyTuple_GET_ITEM(kwnames, i);
      if (PyDict_SetItem(kwargs_dict.get(), key, args[nargs + i]) < 0) {
        return nullptr;
      }
    }
  }

  return PyObject_Call(
      format_node_helper.get(),
      call_args.get(),
      kwargs_dict ? kwargs_dict.get() : nullptr);
}

// __repr__: Return _repr_fn(self) if set, otherwise self.name
static PyObject* NodeBase_repr(PyObject* self) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  if (node->_repr_fn && node->_repr_fn != Py_None) {
    return PyObject_CallOneArg(node->_repr_fn, self);
  }
  return Py_NewRef(node->name);
}

// __setattr__: Custom attribute setting with hooks for name changes
// and _find_nodes_lookup_table updates
static int NodeBase_setattro(
    PyObject* self,
    PyObject* name_obj,
    PyObject* value) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);

  // Skip hooks if graph is not yet initialized
  if (!node->graph) {
    return PyObject_GenericSetAttr(self, name_obj, value);
  }

  const char* attr_name = PyUnicode_AsUTF8(name_obj);
  if (!attr_name) {
    return -1;
  }

  bool is_name = strcmp(attr_name, "name") == 0;

  // Handle "name" attribute change - call replace hooks
  if (is_name && node->name) {
    PyObject* owning_module = GraphBase_borrow_owning_module(node->graph);
    if (owning_module && owning_module != Py_None) {
      THPObjectPtr replace_hooks(
          PyObject_GetAttrString(owning_module, "_replace_hooks"));
      if (replace_hooks && replace_hooks.get() != Py_None &&
          PySequence_Check(replace_hooks)) {
        if (!PyUnicode_Check(value)) {
          PyErr_SetString(PyExc_TypeError, "name must be a string");
          return -1;
        }
        Py_ssize_t num_hooks = PySequence_Size(replace_hooks);
        // Iterate over users and call each hook
        PyObject *user_key = nullptr, *user_value = nullptr;
        Py_ssize_t pos = 0;
        while (PyDict_Next(node->users, &pos, &user_key, &user_value)) {
          for (Py_ssize_t i = 0; i < num_hooks; i++) {
            THPObjectPtr hook(PySequence_GetItem(replace_hooks, i));
            if (hook &&
                !call_replace_hook_with_keywords(
                    hook.get(), self, value, user_key)) {
              return -1;
            }
          }
        }
      }
    }
    PyErr_Clear();
  }

  // The lookup table indexes on (op, target), so only those attributes
  // need remove/re-insert when changed.
  bool needs_lookup_update =
      strcmp(attr_name, "op") == 0 || strcmp(attr_name, "target") == 0;

  PyObject* find_nodes_lookup = nullptr;
  bool update = false;
  if (needs_lookup_update) {
    find_nodes_lookup = GraphBase_borrow_find_nodes_lookup_table(node->graph);
    if (find_nodes_lookup && find_nodes_lookup != Py_None) {
      if (PyObject_HasAttr(self, name_obj)) {
        if (FindNodesLookupTable_contains_impl(find_nodes_lookup, self)) {
          update = true;
          FindNodesLookupTable_remove_impl(find_nodes_lookup, self);
        }
      }
    }
    PyErr_Clear();
  }

  int result = PyObject_GenericSetAttr(self, name_obj, value);

  // Re-insert into lookup table if needed
  if (update && result == 0) {
    FindNodesLookupTable_insert_impl(find_nodes_lookup, self);
    PyErr_Clear();
  }

  return result;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef NodeBase_methods[] = {
    {"_update_args_kwargs",
     (PyCFunction)(void*)NodeBase__update_args_kwargs,
     METH_FASTCALL,
     "Internal method: do not call directly."},
    {"_remove_from_list",
     (PyCFunction)(void*)NodeBase__remove_from_list,
     METH_NOARGS,
     "Internal method: do not call directly."},
    {"_replace_input_with",
     (PyCFunction)(void*)NodeBase__replace_input_with,
     METH_FASTCALL,
     "Internal method: replace occurrences of one input Node with another."},
    {"_prepend",
     (PyCFunction)(void*)NodeBase__prepend,
     METH_O,
     "Internal method: do not call directly."},
    {"__lt__",
     (PyCFunction)(void*)NodeBase___lt__,
     METH_O,
     "Return True if self.sort_key < other.sort_key"},
    {"__gt__",
     (PyCFunction)(void*)NodeBase___gt__,
     METH_O,
     "Return True if self.sort_key > other.sort_key"},
    {"__ge__",
     (PyCFunction)(void*)NodeBase___ge__,
     METH_O,
     "Return True if self.sort_key >= other.sort_key"},
    {"__le__",
     (PyCFunction)(void*)NodeBase___le__,
     METH_O,
     "Return True if self.sort_key <= other.sort_key"},
    {"prepend",
     (PyCFunction)(void*)NodeBase_prepend,
     METH_O,
     "Insert x before this node in the list of nodes in the graph."},
    {"append",
     (PyCFunction)(void*)NodeBase_append,
     METH_O,
     "Insert x after this node in the list of nodes in the graph."},
    {"update_arg",
     (PyCFunction)(void*)NodeBase_update_arg,
     METH_FASTCALL,
     "Update an existing positional argument."},
    {"insert_arg",
     (PyCFunction)(void*)NodeBase_insert_arg,
     METH_FASTCALL,
     "Insert a positional argument at the given index."},
    {"update_kwarg",
     (PyCFunction)(void*)NodeBase_update_kwarg,
     METH_FASTCALL,
     "Update an existing keyword argument."},
    {"replace_input_with",
     (PyCFunction)(void*)NodeBase_replace_input_with,
     METH_FASTCALL,
     "Loop through input nodes and replace all instances of old_input with new_input."},
    {"replace_all_uses_with",
     (PyCFunction)(void*)NodeBase_replace_all_uses_with,
     METH_FASTCALL | METH_KEYWORDS,
     "Replace all uses of self in the Graph with replace_with."},
    {"is_impure",
     (PyCFunction)(void*)NodeBase_is_impure,
     METH_FASTCALL | METH_KEYWORDS,
     "Returns whether this op is impure."},
    {"_rename",
     (PyCFunction)(void*)NodeBase__rename,
     METH_O,
     "Update name through graph namespace."},
    {"__getstate__",
     (PyCFunction)(void*)NodeBase_getstate,
     METH_NOARGS,
     "Return state for pickling."},
    {"__setstate__",
     (PyCFunction)(void*)NodeBase_setstate,
     METH_O,
     "Restore from pickled state."},
    {"normalized_arguments",
     (PyCFunction)(void*)NodeBase_normalized_arguments,
     METH_FASTCALL | METH_KEYWORDS,
     "Returns normalized arguments to Python targets."},
    {"format_node",
     (PyCFunction)(void*)NodeBase_format_node,
     METH_FASTCALL | METH_KEYWORDS,
     "Return a descriptive string representation of self."},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

// Property getter for 'next'
static PyObject* NodeBase_get_next(PyObject* self, void* /*closure*/) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  return Py_NewRef(reinterpret_cast<PyObject*>(node->_next));
}

// Property getter for 'prev'
static PyObject* NodeBase_get_prev(PyObject* self, void* /*closure*/) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  return Py_NewRef(reinterpret_cast<PyObject*>(node->_prev));
}

// Property getter for 'args'
static PyObject* NodeBase_get_args(PyObject* self, void* /*closure*/) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  if (node->_args) {
    return Py_NewRef(node->_args);
  }
  Py_RETURN_NONE;
}

// Property setter for 'args'
static int NodeBase_set_args(
    PyObject* self,
    PyObject* value,
    void* /*closure*/) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  PyObject* update_args[2] = {value, node->_kwargs};
  THPObjectPtr result(NodeBase__update_args_kwargs(self, update_args, 2));
  if (!result) {
    return -1;
  }
  return 0;
}

// Property getter for 'kwargs'
static PyObject* NodeBase_get_kwargs(PyObject* self, void* /*closure*/) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  if (node->_kwargs) {
    return Py_NewRef(node->_kwargs);
  }
  Py_RETURN_NONE;
}

// Property setter for 'kwargs'
static int NodeBase_set_kwargs(
    PyObject* self,
    PyObject* value,
    void* /*closure*/) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  PyObject* update_args[2] = {node->_args, value};
  THPObjectPtr result(NodeBase__update_args_kwargs(self, update_args, 2));
  if (!result) {
    return -1;
  }
  return 0;
}

// Property getter for 'all_input_nodes'
static PyObject* NodeBase_get_all_input_nodes(
    PyObject* self,
    void* /*closure*/) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  return PyDict_Keys(node->_input_nodes);
}

// Property getter for 'stack_trace'
static PyObject* NodeBase_get_stack_trace(PyObject* self, void* /*closure*/) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  PyObject* stack_trace = PyDict_GetItemString(node->meta, "stack_trace");
  if (stack_trace) {
    return Py_NewRef(stack_trace);
  }
  Py_RETURN_NONE;
}

// Property setter for 'stack_trace'
static int NodeBase_set_stack_trace(
    PyObject* self,
    PyObject* value,
    void* /*closure*/) {
  NodeBase* node = reinterpret_cast<NodeBase*>(self);
  if (PyDict_SetItemString(node->meta, "stack_trace", value) < 0) {
    return -1;
  }
  return 0;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyGetSetDef NodeBase_getset[] = {
    {"_sort_key",
     (getter)NodeBase_get_sort_key,
     (setter)NodeBase_set_sort_key,
     (char*)"The sort key as a tuple of ints",
     nullptr},
    {"next",
     (getter)NodeBase_get_next,
     nullptr, // read-only
     (char*)"Returns the next Node in the linked list of Nodes.",
     nullptr},
    {"prev",
     (getter)NodeBase_get_prev,
     nullptr, // read-only
     (char*)"Returns the previous Node in the linked list of Nodes.",
     nullptr},
    {"args",
     (getter)NodeBase_get_args,
     (setter)NodeBase_set_args,
     (char*)"The tuple of arguments to this Node.",
     nullptr},
    {"kwargs",
     (getter)NodeBase_get_kwargs,
     (setter)NodeBase_set_kwargs,
     (char*)"The dict of keyword arguments to this Node.",
     nullptr},
    {"all_input_nodes",
     (getter)NodeBase_get_all_input_nodes,
     nullptr, // read-only
     (char*)"Return all Nodes that are inputs to this Node.",
     nullptr},
    {"stack_trace",
     (getter)NodeBase_get_stack_trace,
     (setter)NodeBase_set_stack_trace,
     (char*)"Return the Python stack trace that was recorded during tracing.",
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr} // Sentinel
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
    NodeBase_repr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    NodeBase_setattro, /* tp_setattro */
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
    NodeBase_getset, /* tp_getset */
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
  } catch (const PythonError&) {
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
  } catch (const PythonError&) {
    return nullptr; // error should already be set
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef extra_methods[] = {
    {"_fx_map_aggregate",
     (PyCFunction)(void*)py_map_aggregate,
     METH_FASTCALL,
     "Recursively apply a function to every element in an aggregate object."},
    {"_fx_map_arg",
     (PyCFunction)(void*)py_map_arg,
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

////////////////////////////////
// Fast C++ accessors for Node attributes
////////////////////////////////

bool NodeBase_Check(PyObject* obj) {
  return PyObject_TypeCheck(obj, &NodeBaseType);
}

PyObject* NodeBase_borrow_op(PyObject* node) {
  if (!NodeBase_Check(node)) {
    return nullptr;
  }
  return reinterpret_cast<NodeBase*>(node)->op;
}

PyObject* NodeBase_borrow_target(PyObject* node) {
  if (!NodeBase_Check(node)) {
    return nullptr;
  }
  return reinterpret_cast<NodeBase*>(node)->target;
}

PyObject* NodeBase_borrow_name(PyObject* node) {
  if (!NodeBase_Check(node)) {
    return nullptr;
  }
  return reinterpret_cast<NodeBase*>(node)->name;
}

PyObject* NodeBase_borrow_graph(PyObject* node) {
  if (!NodeBase_Check(node)) {
    return nullptr;
  }
  return reinterpret_cast<NodeBase*>(node)->graph;
}

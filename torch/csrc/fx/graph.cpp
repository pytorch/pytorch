#include <torch/csrc/fx/graph.h>
#include <torch/csrc/fx/node.h>

#include <structmember.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <string>
#include <unordered_map>

// GraphBase struct definition - outside anonymous namespace for accessor
// visibility
struct GraphBase {
  PyObject_HEAD
  // _root: Node - the sentinel node
  PyObject* _root;
  // _len: int - number of nodes in the graph
  Py_ssize_t _len;
  // _find_nodes_lookup_table: _FindNodesLookupTable
  PyObject* _find_nodes_lookup_table;
  // _owning_module: Optional[GraphModule]
  PyObject* _owning_module;
};

namespace {

static inline bool is_alnum_or_underscore(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
      (c >= '0' && c <= '9') || c == '_';
}

static inline bool is_alpha_or_underscore(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

// Equivalent to re.compile("[^0-9a-zA-Z_]+").sub("_", s)
static std::string replace_illegal_chars(const std::string& s) {
  std::string result;
  result.reserve(s.size());
  bool prev_illegal = false;
  for (char c : s) {
    if (is_alnum_or_underscore(c)) {
      result.push_back(c);
      prev_illegal = false;
    } else if (!prev_illegal) {
      result.push_back('_');
      prev_illegal = true;
    }
  }
  return result;
}

// Equivalent to re.compile(r"^([a-zA-Z_][0-9a-zA-Z_]*?)(?:_(\d+))?$").match(s)
// Returns true on match, setting base and (optionally) num_suffix.
// has_num is set to true when a trailing _\d+ suffix was found.
static bool name_regex_match(
    const std::string& s,
    std::string& base,
    long long& num_suffix,
    bool& has_num) {
  if (s.empty() || !is_alpha_or_underscore(s[0])) {
    return false;
  }
  // All characters must be [a-zA-Z0-9_]
  for (char c : s) {
    if (!is_alnum_or_underscore(c)) {
      return false;
    }
  }
  // Find the last '_' that is followed by only digits (the _\d+ suffix).
  // The base part uses a non-greedy match, so we want the *last* such split.
  size_t split = s.rfind('_');
  if (split != std::string::npos && split > 0 && split + 1 < s.size()) {
    bool all_digits = true;
    for (size_t i = split + 1; i < s.size(); ++i) {
      if (s[i] < '0' || s[i] > '9') {
        all_digits = false;
        break;
      }
    }
    if (all_digits) {
      base = s.substr(0, split);
      num_suffix = std::stoll(s.substr(split + 1));
      has_num = true;
      return true;
    }
  }
  base = s;
  num_suffix = 0;
  has_num = false;
  return true;
}

struct NamespaceBase {
  PyObject_HEAD
  // _obj_to_name: dict[Any, str]
  PyObject* obj_to_name;
  // _used_names: set[str]
  PyObject* used_names;
  // _base_count: dict[str, int]
  PyObject* base_count;
};

static int NamespaceBase_clear(NamespaceBase* self);

static PyObject* NamespaceBase_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  PyObject* self = type->tp_alloc(type, 0);
  if (!self)
    return nullptr;
  NamespaceBase* ns = reinterpret_cast<NamespaceBase*>(self);
  ns->obj_to_name = nullptr;
  ns->used_names = nullptr;
  ns->base_count = nullptr;
  return self;
}

static int NamespaceBase_init_fn(
    NamespaceBase* self,
    PyObject* args,
    PyObject* kwds) {
  self->obj_to_name = PyDict_New();
  if (!self->obj_to_name) {
    goto fail;
  }
  self->used_names = PySet_New(nullptr);
  if (!self->used_names) {
    goto fail;
  }
  self->base_count = PyDict_New();
  if (!self->base_count) {
    goto fail;
  }
  return 0;

fail:
  NamespaceBase_clear(self);
  return -1;
}

static int NamespaceBase_traverse(
    NamespaceBase* self,
    visitproc visit,
    void* arg) {
  Py_VISIT(self->obj_to_name);
  Py_VISIT(self->used_names);
  Py_VISIT(self->base_count);
  return 0;
}

static int NamespaceBase_clear(NamespaceBase* self) {
  Py_CLEAR(self->obj_to_name);
  Py_CLEAR(self->used_names);
  Py_CLEAR(self->base_count);
  return 0;
}

static void NamespaceBase_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  (void)NamespaceBase_clear(reinterpret_cast<NamespaceBase*>(self));
  Py_TYPE(self)->tp_free(self);
}

// Get _illegal_names from Python (cached)
static PyObject* get_illegal_names() {
  static PyObject* illegal_names = nullptr;
  if (!illegal_names) {
    THPObjectPtr graph_module(PyImport_ImportModule("torch.fx.graph"));
    if (!graph_module) {
      return nullptr;
    }
    illegal_names =
        PyObject_GetAttrString(graph_module.get(), "_illegal_names");
    if (!illegal_names) {
      return nullptr;
    }
  }
  return illegal_names;
}

// create_name(candidate, obj) -> str
static PyObject* NamespaceBase_create_name(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(
        PyExc_TypeError,
        "create_name() requires exactly 2 arguments (candidate, obj)");
    return nullptr;
  }

  NamespaceBase* ns = reinterpret_cast<NamespaceBase*>(self);
  PyObject* candidate_obj = args[0];
  PyObject* obj = args[1];

  // If obj is not None and already in _obj_to_name, return existing name
  if (obj != Py_None) {
    PyObject* existing = PyDict_GetItem(ns->obj_to_name, obj);
    if (existing) {
      return Py_NewRef(existing);
    }
  }

  // Get candidate as C++ string
  const char* candidate_cstr = PyUnicode_AsUTF8(candidate_obj);
  if (!candidate_cstr) {
    return nullptr;
  }
  std::string candidate(candidate_cstr);

  std::string base;
  long long num = 0;
  bool has_num = false;

  if (!name_regex_match(candidate, base, num, has_num)) {
    // Delete all characters that are illegal in a Python identifier
    candidate = replace_illegal_chars(candidate);

    if (candidate.empty()) {
      candidate = "_unnamed";
    }

    if (candidate[0] >= '0' && candidate[0] <= '9') {
      candidate = "_" + candidate;
    }

    if (!name_regex_match(candidate, base, num, has_num)) {
      PyErr_SetString(PyExc_AssertionError, "Failed to create valid name");
      return nullptr;
    }
  }

  THPObjectPtr candidate_py(PyUnicode_FromString(candidate.c_str()));
  if (!candidate_py) {
    return nullptr;
  }

  // Check if candidate is in used_names (PySet_Contains returns -1 on error)
  int in_used_names = PySet_Contains(ns->used_names, candidate_py.get());
  if (in_used_names < 0) {
    return nullptr;
  }

  if (!has_num || in_used_names) {
    // num = self._base_count.get(candidate, 0)
    PyObject* count_obj = PyDict_GetItem(ns->base_count, candidate_py.get());
    if (count_obj) {
      num = PyLong_AsLongLong(count_obj);
      if (num == -1 && PyErr_Occurred()) {
        return nullptr;
      }
    }

    // Check _illegal_names
    PyObject* illegal_names = get_illegal_names();
    if (!illegal_names) {
      return nullptr;
    }
    PyObject* illegal_val = PyDict_GetItem(illegal_names, candidate_py.get());
    // if _illegal_names.get(candidate, obj) is not obj:
    // When not found, .get() returns the default (obj), so "is not obj" is
    // False. When found, check if the value is different from obj.
    if (illegal_val != nullptr && illegal_val != obj) {
      num += 1;
      candidate = base + "_" + std::to_string(num);
      candidate_py = THPObjectPtr(PyUnicode_FromString(candidate.c_str()));
      if (!candidate_py) {
        return nullptr;
      }
    }
  } else {
    // num is already set by name_regex_match
  }

  // while candidate in self._used_names:
  while (true) {
    int contains = PySet_Contains(ns->used_names, candidate_py.get());
    if (contains < 0) {
      return nullptr; // Error
    }
    if (!contains) {
      break; // Not in set, done
    }
    num += 1;
    candidate = base + "_" + std::to_string(num);
    candidate_py = THPObjectPtr(PyUnicode_FromString(candidate.c_str()));
    if (!candidate_py) {
      return nullptr;
    }
  }

  // self._used_names.add(candidate)
  if (PySet_Add(ns->used_names, candidate_py.get()) < 0) {
    return nullptr;
  }

  // self._base_count[base] = num
  THPObjectPtr base_key(PyUnicode_FromString(base.c_str()));
  if (!base_key) {
    return nullptr;
  }
  THPObjectPtr num_val(PyLong_FromLongLong(num));
  if (!num_val) {
    return nullptr;
  }
  if (PyDict_SetItem(ns->base_count, base_key.get(), num_val.get()) < 0) {
    return nullptr;
  }

  // if obj is not None: self._obj_to_name[obj] = candidate
  if (obj != Py_None) {
    if (PyDict_SetItem(ns->obj_to_name, obj, candidate_py.get()) < 0) {
      return nullptr;
    }
  }

  return candidate_py.release();
}

// associate_name_with_obj(name, obj) -> None
static PyObject* NamespaceBase_associate_name_with_obj(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(
        PyExc_TypeError,
        "associate_name_with_obj() requires exactly 2 arguments (name, obj)");
    return nullptr;
  }

  NamespaceBase* ns = reinterpret_cast<NamespaceBase*>(self);
  PyObject* name = args[0];
  PyObject* obj = args[1];

  // maybe_existing = self._obj_to_name.setdefault(obj, name)
  PyObject* existing = PyDict_SetDefault(ns->obj_to_name, obj, name);
  if (!existing) {
    return nullptr;
  }

  // assert maybe_existing is name
  if (existing != name) {
    PyErr_SetString(PyExc_AssertionError, "obj is already associated");
    return nullptr;
  }

  Py_RETURN_NONE;
}

// _rename_object(obj, name) -> None
static PyObject* NamespaceBase_rename_object(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(
        PyExc_TypeError,
        "_rename_object() requires exactly 2 arguments (obj, name)");
    return nullptr;
  }

  NamespaceBase* ns = reinterpret_cast<NamespaceBase*>(self);
  PyObject* obj = args[0];
  PyObject* name = args[1];

  // assert obj in self._obj_to_name
  int contains = PyDict_Contains(ns->obj_to_name, obj);
  if (contains < 0) {
    return nullptr;
  }
  if (!contains) {
    PyErr_SetString(PyExc_AssertionError, "obj not in _obj_to_name");
    return nullptr;
  }

  // self._obj_to_name[obj] = name
  if (PyDict_SetItem(ns->obj_to_name, obj, name) < 0) {
    return nullptr;
  }

  // self._used_names.add(name)
  if (PySet_Add(ns->used_names, name) < 0) {
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyObject* NamespaceBase_getstate(
    PyObject* self,
    PyObject* Py_UNUSED(ignored)) {
  NamespaceBase* ns = reinterpret_cast<NamespaceBase*>(self);
  return Py_BuildValue(
      "(OOO)", ns->obj_to_name, ns->used_names, ns->base_count);
}

static PyObject* NamespaceBase_setstate(PyObject* self, PyObject* state) {
  NamespaceBase* ns = reinterpret_cast<NamespaceBase*>(self);
  PyObject *obj_to_name = nullptr, *used_names = nullptr, *base_count = nullptr;
  if (!PyArg_ParseTuple(state, "OOO", &obj_to_name, &used_names, &base_count))
    return nullptr;
  Py_XDECREF(ns->obj_to_name);
  Py_XDECREF(ns->used_names);
  Py_XDECREF(ns->base_count);
  ns->obj_to_name = Py_NewRef(obj_to_name);
  ns->used_names = Py_NewRef(used_names);
  ns->base_count = Py_NewRef(base_count);
  Py_RETURN_NONE;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef NamespaceBase_methods[] = {
    {"create_name",
     (PyCFunction)(void*)NamespaceBase_create_name,
     METH_FASTCALL,
     "Create a unique name."},
    {"associate_name_with_obj",
     (PyCFunction)(void*)NamespaceBase_associate_name_with_obj,
     METH_FASTCALL,
     "Associate a unique name with an object."},
    {"_rename_object",
     (PyCFunction)(void*)NamespaceBase_rename_object,
     METH_FASTCALL,
     "Update the name associated with an object."},
    {"__getstate__",
     NamespaceBase_getstate,
     METH_NOARGS,
     "Return state for pickling."},
    {"__setstate__",
     NamespaceBase_setstate,
     METH_O,
     "Restore state from pickling."},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMemberDef NamespaceBase_members[] = {
    {"_obj_to_name",
     T_OBJECT_EX,
     offsetof(NamespaceBase, obj_to_name),
     0,
     "Dict mapping objects to their names"},
    {"_used_names",
     T_OBJECT_EX,
     offsetof(NamespaceBase, used_names),
     0,
     "Set of all used names"},
    {"_base_count",
     T_OBJECT_EX,
     offsetof(NamespaceBase, base_count),
     0,
     "Dict mapping base names to their current count"},
    {nullptr, 0, 0, 0, nullptr} // Sentinel
};

static PyTypeObject NamespaceBaseType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._NamespaceBase", /* tp_name */
    sizeof(NamespaceBase), /* tp_basicsize */
    0, /* tp_itemsize */
    NamespaceBase_dealloc, /* tp_dealloc */
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
    "A context for associating names uniquely with objects.", /* tp_doc */
    (traverseproc)NamespaceBase_traverse, /* tp_traverse */
    (inquiry)NamespaceBase_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    NamespaceBase_methods, /* tp_methods */
    NamespaceBase_members, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)NamespaceBase_init_fn, /* tp_init */
    nullptr, /* tp_alloc */
    NamespaceBase_new, /* tp_new */
};

////////////////////////////////
// _FindNodesLookupTable
////////////////////////////////

// Hash function for (PyObject* op_str, PyObject* target) pairs
// Uses value-based hashing and comparison via Python's __hash__ and __eq__
struct LookupKey {
  PyObject* op;
  PyObject* target; // can be nullptr for non-call_function ops

  bool operator==(const LookupKey& other) const {
    // For op, compare strings by value
    if (PyUnicode_Compare(op, other.op) != 0 || (PyErr_Occurred() != nullptr)) {
      PyErr_Clear();
      return false;
    }
    // For target, use value comparison via __eq__
    // This is needed for custom extension ops that define __eq__
    if (target == other.target) {
      return true; // Fast path for identity
    }
    if (target == nullptr || other.target == nullptr) {
      return target == other.target;
    }
    int result = PyObject_RichCompareBool(target, other.target, Py_EQ);
    if (result < 0) {
      PyErr_Clear();
      return false;
    }
    return result != 0;
  }
};

struct LookupKeyHash {
  size_t operator()(const LookupKey& key) const {
    // Hash the op string by value
    Py_hash_t op_hash = PyObject_Hash(key.op);
    if (op_hash == -1) {
      PyErr_Clear();
      op_hash = 0;
    }
    // Hash the target by value via __hash__
    // This is needed for custom extension ops that define __hash__
    Py_hash_t target_hash = 0;
    if (key.target != nullptr) {
      target_hash = PyObject_Hash(key.target);
      if (target_hash == -1) {
        PyErr_Clear();
        target_hash = 0;
      }
    }
    // Combine hashes
    return static_cast<size_t>(op_hash) ^
        (static_cast<size_t>(target_hash) << 1);
  }
};

struct FindNodesLookupTable {
  PyObject_HEAD
  // table: dict[(op, target), dict[Node, None]]
  // For call_function: key is (op, target)
  // For other ops: key is (op, None)
  // Using unordered_map for the outer dict, PyObject* (dict) for inner
  std::unordered_map<LookupKey, PyObject*, LookupKeyHash>* table;
};

static int FindNodesLookupTable_clear(FindNodesLookupTable* self);

static PyObject* FindNodesLookupTable_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  PyObject* self = type->tp_alloc(type, 0);
  if (!self)
    return nullptr;
  FindNodesLookupTable* lt = reinterpret_cast<FindNodesLookupTable*>(self);
  lt->table = nullptr;
  return self;
}

static int FindNodesLookupTable_init_fn(
    FindNodesLookupTable* self,
    PyObject* args,
    PyObject* kwds) {
  self->table = new std::unordered_map<LookupKey, PyObject*, LookupKeyHash>();
  return 0;
}

static int FindNodesLookupTable_traverse(
    FindNodesLookupTable* self,
    visitproc visit,
    void* arg) {
  if (self->table) {
    for (auto& pair : *self->table) {
      Py_VISIT(pair.first.op);
      Py_VISIT(pair.first.target);
      Py_VISIT(pair.second);
    }
  }
  return 0;
}

static int FindNodesLookupTable_clear(FindNodesLookupTable* self) {
  if (self->table) {
    for (auto& pair : *self->table) {
      // Decref the key's op and target (which were incref'd when inserted)
      Py_XDECREF(pair.first.op);
      Py_XDECREF(pair.first.target);
      // Decref the inner dict
      Py_CLEAR(pair.second);
    }
    delete self->table;
    self->table = nullptr;
  }
  return 0;
}

static void FindNodesLookupTable_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  (void)FindNodesLookupTable_clear(
      reinterpret_cast<FindNodesLookupTable*>(self));
  Py_TYPE(self)->tp_free(self);
}

// Cache "call_function" string
static PyObject* get_call_function_str() {
  static PyObject* call_function_str = nullptr;
  if (!call_function_str) {
    call_function_str = PyUnicode_InternFromString("call_function");
  }
  return call_function_str;
}

// RAII wrapper for LookupKey that manages reference counting
// Used for temporary keys during lookup operations
struct OwnedLookupKey {
  THPObjectPtr op;
  THPObjectPtr target;

  OwnedLookupKey() : op(nullptr), target(nullptr) {}

  // Create from node - returns empty key on error
  // Uses fast C++ accessors (node must be a NodeBase instance)
  static OwnedLookupKey from_node(PyObject* node) {
    OwnedLookupKey result;

    PyObject* op = NodeBase_borrow_op(node);
    if (!op) {
      return {};
    }
    result.op = THPObjectPtr(Py_NewRef(op));

    PyObject* call_function = get_call_function_str();
    if (PyUnicode_Compare(op, call_function) == 0 && !PyErr_Occurred()) {
      PyObject* target = NodeBase_borrow_target(node);
      if (target) {
        result.target = THPObjectPtr(Py_NewRef(target));
      }
    }
    PyErr_Clear();
    return result;
  }

  bool valid() const {
    return op.get() != nullptr;
  }

  // Convert to LookupKey for map lookup (borrows references)
  LookupKey as_lookup_key() {
    LookupKey key;
    key.op = op.get();
    key.target = target.get();
    return key;
  }

  // Release ownership and return LookupKey (for storing in map)
  LookupKey release() {
    LookupKey key;
    key.op = op.release();
    key.target = target.release();
    return key;
  }
};

// __contains__(node) -> bool
static int FindNodesLookupTable_contains(PyObject* self, PyObject* node) {
  FindNodesLookupTable* lt = reinterpret_cast<FindNodesLookupTable*>(self);
  OwnedLookupKey key = OwnedLookupKey::from_node(node);
  if (!key.valid()) {
    return -1;
  }

  auto it = lt->table->find(key.as_lookup_key());
  if (it == lt->table->end()) {
    return 0;
  }

  return PyDict_Contains(it->second, node);
}

// insert(node) -> None
static PyObject* FindNodesLookupTable_insert(PyObject* self, PyObject* node) {
  FindNodesLookupTable* lt = reinterpret_cast<FindNodesLookupTable*>(self);
  OwnedLookupKey key = OwnedLookupKey::from_node(node);
  if (!key.valid()) {
    return nullptr;
  }

  auto it = lt->table->find(key.as_lookup_key());
  PyObject* inner_dict;
  if (it == lt->table->end()) {
    // Create new inner dict and insert
    inner_dict = PyDict_New();
    if (!inner_dict) {
      return nullptr;
    }
    // Release ownership - key references are transferred to the table
    (*lt->table)[key.release()] = inner_dict;
  } else {
    inner_dict = it->second;
    // Key will be automatically cleaned up by OwnedLookupKey destructor
  }

  if (PyDict_SetItem(inner_dict, node, Py_None) < 0) {
    return nullptr;
  }

  Py_RETURN_NONE;
}

// remove(node) -> None
static PyObject* FindNodesLookupTable_remove(PyObject* self, PyObject* node) {
  FindNodesLookupTable* lt = reinterpret_cast<FindNodesLookupTable*>(self);
  OwnedLookupKey key = OwnedLookupKey::from_node(node);
  if (!key.valid()) {
    return nullptr;
  }

  auto it = lt->table->find(key.as_lookup_key());
  // Key will be automatically cleaned up by OwnedLookupKey destructor

  if (it != lt->table->end()) {
    if (PyDict_DelItem(it->second, node) < 0) {
      PyErr_Clear(); // Ignore KeyError if node not found
    }
  }

  Py_RETURN_NONE;
}

// find_nodes(*, op: str, target: Optional[Target] = None) -> list[Node]
static PyObject* FindNodesLookupTable_find_nodes(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames) {
  FindNodesLookupTable* lt = reinterpret_cast<FindNodesLookupTable*>(self);

  // Parse keyword arguments
  PyObject* op = nullptr;
  PyObject* target = Py_None;

  if (nargs > 0) {
    PyErr_SetString(
        PyExc_TypeError, "find_nodes() takes no positional arguments");
    return nullptr;
  }

  if (kwnames) {
    Py_ssize_t nkwargs = PyTuple_GET_SIZE(kwnames);
    for (Py_ssize_t i = 0; i < nkwargs; i++) {
      PyObject* key = PyTuple_GET_ITEM(kwnames, i);
      const char* key_str = PyUnicode_AsUTF8(key);
      if (!key_str)
        return nullptr;
      if (strcmp(key_str, "op") == 0) {
        op = args[i];
      } else if (strcmp(key_str, "target") == 0) {
        target = args[i];
      } else {
        PyErr_Format(
            PyExc_TypeError,
            "find_nodes() got unexpected keyword argument '%s'",
            key_str);
        return nullptr;
      }
    }
  }

  if (!op) {
    PyErr_SetString(
        PyExc_TypeError, "find_nodes() missing required argument: 'op'");
    return nullptr;
  }

  PyObject* call_function = get_call_function_str();
  int is_call_function =
      (PyUnicode_Compare(op, call_function) == 0 && !PyErr_Occurred());
  if (PyErr_Occurred()) {
    return nullptr;
  }

  if (is_call_function) {
    // For call_function, target is required and we do exact lookup
    if (target == Py_None) {
      PyErr_SetString(
          PyExc_AssertionError,
          "target is required for call_function operations");
      return nullptr;
    }

    LookupKey key = {op, target};
    auto it = lt->table->find(key);
    if (it == lt->table->end()) {
      return PyList_New(0);
    }
    return PyDict_Keys(it->second);
  }

  // For other ops, lookup with (op, None)
  LookupKey key = {op, nullptr};
  auto it = lt->table->find(key);
  if (it == lt->table->end()) {
    return PyList_New(0);
  }

  if (target == Py_None) {
    // No target filter, return all nodes
    return PyDict_Keys(it->second);
  }

  // Filter by target
  THPObjectPtr result(PyList_New(0));
  if (!result) {
    return nullptr;
  }

  PyObject *node_key = nullptr, *node_val = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(it->second, &pos, &node_key, &node_val)) {
    PyObject* node_target = NodeBase_borrow_target(node_key);
    if (!node_target) {
      continue; // Skip non-NodeBase objects
    }
    int eq = PyObject_RichCompareBool(node_target, target, Py_EQ);
    if (eq < 0) {
      return nullptr;
    }
    if (eq) {
      if (PyList_Append(result.get(), node_key) < 0) {
        return nullptr;
      }
    }
  }

  return result.release();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef FindNodesLookupTable_methods[] = {
    {"insert",
     (PyCFunction)FindNodesLookupTable_insert,
     METH_O,
     "Insert a node into the lookup table."},
    {"remove",
     (PyCFunction)FindNodesLookupTable_remove,
     METH_O,
     "Remove a node from the lookup table."},
    {"find_nodes",
     (PyCFunction)(void*)FindNodesLookupTable_find_nodes,
     METH_FASTCALL | METH_KEYWORDS,
     "Find nodes by op and optionally target."},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

static PySequenceMethods FindNodesLookupTable_as_sequence = {
    nullptr, /* sq_length */
    nullptr, /* sq_concat */
    nullptr, /* sq_repeat */
    nullptr, /* sq_item */
    nullptr, /* sq_slice */
    nullptr, /* sq_ass_item */
    nullptr, /* sq_ass_slice */
    FindNodesLookupTable_contains, /* sq_contains */
    nullptr, /* sq_inplace_concat */
    nullptr, /* sq_inplace_repeat */
};

static PyTypeObject FindNodesLookupTableType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._FindNodesLookupTable", /* tp_name */
    sizeof(FindNodesLookupTable), /* tp_basicsize */
    0, /* tp_itemsize */
    FindNodesLookupTable_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    &FindNodesLookupTable_as_sequence, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /* tp_flags */
    "Side table for the graph for doing fast queries.", /* tp_doc */
    (traverseproc)FindNodesLookupTable_traverse, /* tp_traverse */
    (inquiry)FindNodesLookupTable_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    FindNodesLookupTable_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)FindNodesLookupTable_init_fn, /* tp_init */
    nullptr, /* tp_alloc */
    FindNodesLookupTable_new, /* tp_new */
};

////////////////////////////////
// _node_list
////////////////////////////////

struct NodeList {
  PyObject_HEAD
  PyObject* graph;
  bool reversed;
};

static int NodeList_clear(NodeList* self);

static PyObject* NodeList_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  PyObject* self = type->tp_alloc(type, 0);
  if (!self)
    return nullptr;
  NodeList* nl = reinterpret_cast<NodeList*>(self);
  nl->graph = nullptr;
  nl->reversed = false;
  return self;
}

static int NodeList_init_fn(NodeList* self, PyObject* args, PyObject* kwds) {
  PyObject* graph = nullptr;
  const char* direction = "_next";
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* keywords[] = {"graph", "direction", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwds,
          "O|s",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(keywords),
          &graph,
          &direction)) {
    return -1;
  }

  self->graph = Py_NewRef(graph);
  self->reversed = (strcmp(direction, "_prev") == 0);
  // Note: Do NOT call PyObject_GC_Track here - Python handles GC tracking
  // automatically for types with Py_TPFLAGS_HAVE_GC.
  return 0;
}

static int NodeList_traverse(NodeList* self, visitproc visit, void* arg) {
  Py_VISIT(self->graph);
  return 0;
}

static int NodeList_clear(NodeList* self) {
  Py_CLEAR(self->graph);
  return 0;
}

static void NodeList_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  (void)NodeList_clear(reinterpret_cast<NodeList*>(self));
  Py_TYPE(self)->tp_free(self);
}

// __len__
static Py_ssize_t NodeList_len(PyObject* self) {
  NodeList* nl = reinterpret_cast<NodeList*>(self);
  if (GraphBase_Check(nl->graph)) {
    return reinterpret_cast<GraphBase*>(nl->graph)->_len;
  }
  PyObject* len_obj = PyObject_GetAttrString(nl->graph, "_len");
  if (!len_obj) {
    return -1;
  }
  Py_ssize_t len = PyLong_AsSsize_t(len_obj);
  Py_DECREF(len_obj);
  return len;
}

static PyObject* get_node_iter_type() {
  static PyObject* node_iter_type = nullptr;
  if (!node_iter_type) {
    THPObjectPtr torch_c(PyImport_ImportModule("torch._C"));
    if (!torch_c) {
      return nullptr;
    }
    node_iter_type = PyObject_GetAttrString(torch_c.get(), "_NodeIter");
  }
  return node_iter_type;
}

// __iter__
static PyObject* NodeList_iter(PyObject* self) {
  NodeList* nl = reinterpret_cast<NodeList*>(self);
  PyObject* root = PyObject_GetAttrString(nl->graph, "_root");
  if (!root) {
    return nullptr;
  }

  PyObject* node_iter_type = get_node_iter_type();
  if (!node_iter_type) {
    Py_DECREF(root);
    return nullptr;
  }

  // Create the iterator: _NodeIter(root, reversed)
  THPObjectPtr args(Py_BuildValue("(Oi)", root, nl->reversed ? 1 : 0));
  Py_DECREF(root);
  if (!args) {
    return nullptr;
  }
  return PyObject_Call(node_iter_type, args.get(), nullptr);
}

// Forward declaration - defined after NodeListType
static PyObject* NodeList_reversed(PyObject* self, PyObject* Py_UNUSED(args));

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef NodeList_methods[] = {
    {"__reversed__",
     (PyCFunction)NodeList_reversed,
     METH_NOARGS,
     "Return a reversed iterator."},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

static PySequenceMethods NodeList_as_sequence = {
    NodeList_len, /* sq_length */
    nullptr, /* sq_concat */
    nullptr, /* sq_repeat */
    nullptr, /* sq_item */
    nullptr, /* sq_slice */
    nullptr, /* sq_ass_item */
    nullptr, /* sq_ass_slice */
    nullptr, /* sq_contains */
    nullptr, /* sq_inplace_concat */
    nullptr, /* sq_inplace_repeat */
};

static PyTypeObject NodeListType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._node_list", /* tp_name */
    sizeof(NodeList), /* tp_basicsize */
    0, /* tp_itemsize */
    NodeList_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    &NodeList_as_sequence, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /* tp_flags */
    "A list-like view of nodes in a graph.", /* tp_doc */
    (traverseproc)NodeList_traverse, /* tp_traverse */
    (inquiry)NodeList_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    NodeList_iter, /* tp_iter */
    nullptr, /* tp_iternext */
    NodeList_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)NodeList_init_fn, /* tp_init */
    nullptr, /* tp_alloc */
    NodeList_new, /* tp_new */
};

// __reversed__
static PyObject* NodeList_reversed(PyObject* self, PyObject* Py_UNUSED(args)) {
  NodeList* nl = reinterpret_cast<NodeList*>(self);
  const char* new_direction = nl->reversed ? "_next" : "_prev";
  THPObjectPtr args(Py_BuildValue("(Os)", nl->graph, new_direction));
  if (!args) {
    return nullptr;
  }
  return PyObject_Call((PyObject*)&NodeListType, args.get(), nullptr);
}

////////////////////////////////
// GraphBase
////////////////////////////////

static int GraphBase_clear(GraphBase* self);

static PyObject* GraphBase_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  PyObject* self = type->tp_alloc(type, 0);
  if (!self)
    return nullptr;
  GraphBase* gb = reinterpret_cast<GraphBase*>(self);
  gb->_root = nullptr;
  gb->_len = 0;
  gb->_find_nodes_lookup_table = nullptr;
  gb->_owning_module = nullptr;
  return self;
}

static int GraphBase_init_fn(GraphBase* self, PyObject* args, PyObject* kwds) {
  // Create the _find_nodes_lookup_table
  self->_find_nodes_lookup_table =
      PyObject_CallNoArgs((PyObject*)&FindNodesLookupTableType);
  if (!self->_find_nodes_lookup_table) {
    return -1;
  }

  self->_len = 0;
  // _root will be set by the Python subclass after __init__ returns
  self->_root = nullptr;

  // Note: Do NOT call PyObject_GC_Track here - the Python subclass
  // (Graph) handles GC tracking through its own object management.
  return 0;
}

static int GraphBase_traverse(GraphBase* self, visitproc visit, void* arg) {
  Py_VISIT(self->_root);
  Py_VISIT(self->_find_nodes_lookup_table);
  Py_VISIT(self->_owning_module);
  return 0;
}

static int GraphBase_clear(GraphBase* self) {
  Py_CLEAR(self->_root);
  Py_CLEAR(self->_find_nodes_lookup_table);
  Py_CLEAR(self->_owning_module);
  return 0;
}

static void GraphBase_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  (void)GraphBase_clear(reinterpret_cast<GraphBase*>(self));
  Py_TYPE(self)->tp_free(self);
}

// Property getter for 'nodes'
static PyObject* GraphBase_get_nodes(PyObject* self, void* /*closure*/) {
  // Create a new _node_list with this graph
  THPObjectPtr args(Py_BuildValue("(O)", self));
  if (!args) {
    return nullptr;
  }
  return PyObject_Call((PyObject*)&NodeListType, args.get(), nullptr);
}

// Property getter for 'owning_module'
static PyObject* GraphBase_get_owning_module(
    PyObject* self,
    void* /*closure*/) {
  GraphBase* gb = reinterpret_cast<GraphBase*>(self);
  if (gb->_owning_module) {
    return Py_NewRef(gb->_owning_module);
  }
  Py_RETURN_NONE;
}

// Property setter for 'owning_module'
static int GraphBase_set_owning_module(
    PyObject* self,
    PyObject* value,
    void* /*closure*/) {
  GraphBase* gb = reinterpret_cast<GraphBase*>(self);
  PyObject* old = gb->_owning_module;
  if (value == Py_None) {
    gb->_owning_module = nullptr;
  } else {
    gb->_owning_module = Py_NewRef(value);
  }
  Py_XDECREF(old);
  return 0;
}

// __getstate__: serialize Graph for pickling
// Saves the linked list of nodes as a flat list of live Node objects.
// Node.__getstate__ omits graph/_prev/_next (breaking circular refs),
// so pickle can serialize the nodes without pulling in the whole graph.
// The lookup table is rebuilt on restore.
static PyObject* GraphBase_getstate(
    PyObject* self,
    PyObject* Py_UNUSED(ignored)) {
  GraphBase* gb = reinterpret_cast<GraphBase*>(self);

  THPObjectPtr dict(PyDict_New());
  if (!dict) {
    return nullptr;
  }

  // Merge the Python subclass __dict__
  THPObjectPtr instance_dict(PyObject_GetAttrString(self, "__dict__"));
  if (instance_dict) {
    if (PyDict_Update(dict.get(), instance_dict.get()) < 0) {
      return nullptr;
    }
  }
  PyErr_Clear();

  // Save _owning_module
  PyObject* om = gb->_owning_module ? gb->_owning_module : Py_None;
  if (PyDict_SetItemString(dict.get(), "_owning_module", om) < 0)
    return nullptr;

  // Collect all nodes (including _root) as live objects in linked-list order.
  // Element 0 is the root sentinel; real nodes follow.
  if (!gb->_root) {
    PyErr_SetString(PyExc_RuntimeError, "Graph has no _root node");
    return nullptr;
  }

  THPObjectPtr nodes(PyList_New(0));
  if (!nodes) {
    return nullptr;
  }
  if (PyList_Append(nodes.get(), gb->_root) < 0)
    return nullptr;

  THPObjectPtr cur(PyObject_GetAttrString(gb->_root, "_next"));
  if (!cur) {
    return nullptr;
  }
  while (cur.get() != gb->_root) {
    if (PyList_Append(nodes.get(), cur.get()) < 0)
      return nullptr;
    THPObjectPtr next(PyObject_GetAttrString(cur.get(), "_next"));
    if (!next) {
      return nullptr;
    }
    cur = std::move(next);
  }

  if (PyDict_SetItemString(dict.get(), "_nodes", nodes.get()) < 0)
    return nullptr;

  return dict.release();
}

// __setstate__: restore Graph from pickled state
// _nodes contains live (already-unpickled) Node objects.  We set their
// graph back-pointer, relink _prev/_next, and rebuild the lookup table.
static PyObject* GraphBase_setstate(PyObject* self, PyObject* state) {
  if (!PyDict_Check(state)) {
    PyErr_SetString(PyExc_TypeError, "state must be a dict");
    return nullptr;
  }
  GraphBase* gb = reinterpret_cast<GraphBase*>(self);

  // Extract the special keys we handle ourselves
  THPObjectPtr nodes(PyMapping_GetItemString(state, "_nodes"));
  if (!nodes || !PyList_Check(nodes.get())) {
    PyErr_SetString(PyExc_TypeError, "state must contain a '_nodes' list");
    return nullptr;
  }

  PyObject* owning_module = nullptr;
  THPObjectPtr om_holder(PyMapping_GetItemString(state, "_owning_module"));
  if (om_holder) {
    owning_module = om_holder.get();
  } else {
    PyErr_Clear();
  }

  // Restore _owning_module
  PyObject* old_om = gb->_owning_module;
  if (owning_module && owning_module != Py_None) {
    gb->_owning_module = Py_NewRef(owning_module);
  } else {
    gb->_owning_module = nullptr;
  }
  Py_XDECREF(old_om);

  // Restore remaining dict items into self.__dict__
  PyObject *key = nullptr, *value = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(state, &pos, &key, &value)) {
    const char* key_str = PyUnicode_AsUTF8(key);
    if (!key_str) {
      return nullptr;
    }
    if (strcmp(key_str, "_nodes") == 0 ||
        strcmp(key_str, "_owning_module") == 0) {
      continue;
    }
    if (PyObject_SetAttr(self, key, value) < 0) {
      return nullptr;
    }
  }

  // _nodes[0] is the root sentinel; remaining elements are real nodes.
  Py_ssize_t n = PyList_GET_SIZE(nodes.get());
  if (n == 0) {
    PyErr_SetString(PyExc_ValueError, "_nodes list must not be empty");
    return nullptr;
  }

  // Create a fresh lookup table
  THPObjectPtr new_lt(
      PyObject_CallNoArgs((PyObject*)&FindNodesLookupTableType));
  if (!new_lt) {
    return nullptr;
  }
  Py_XDECREF(gb->_find_nodes_lookup_table);
  gb->_find_nodes_lookup_table = new_lt.release();

  // Pre-intern attribute names used in the loop
  static PyObject* graph_str = PyUnicode_InternFromString("graph");
  static PyObject* prev_str = PyUnicode_InternFromString("_prev");
  static PyObject* next_str = PyUnicode_InternFromString("_next");

  // Set graph back-pointer, relink _prev/_next, and insert into lookup table
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* node = PyList_GET_ITEM(nodes.get(), i);
    PyObject* prev = PyList_GET_ITEM(nodes.get(), (i - 1 + n) % n);
    PyObject* next = PyList_GET_ITEM(nodes.get(), (i + 1) % n);
    if (PyObject_SetAttr(node, graph_str, self) < 0)
      return nullptr;
    if (PyObject_SetAttr(node, prev_str, prev) < 0)
      return nullptr;
    if (PyObject_SetAttr(node, next_str, next) < 0)
      return nullptr;
    if (i > 0) {
      if (!FindNodesLookupTable_insert_impl(
              gb->_find_nodes_lookup_table, node)) {
        PyErr_SetString(
            PyExc_RuntimeError, "Failed to insert node into lookup table");
        return nullptr;
      }
    }
  }

  // Set _root and _len
  Py_XDECREF(gb->_root);
  gb->_root = Py_NewRef(PyList_GET_ITEM(nodes.get(), 0));
  gb->_len = n - 1; // exclude root

  Py_RETURN_NONE;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef GraphBase_methods[] = {
    {"__getstate__",
     GraphBase_getstate,
     METH_NOARGS,
     "Return state for pickling."},
    {"__setstate__",
     GraphBase_setstate,
     METH_O,
     "Restore state from pickling."},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMemberDef GraphBase_members[] = {
    {"_root", T_OBJECT_EX, offsetof(GraphBase, _root), 0, "The root node"},
    {"_len", T_PYSSIZET, offsetof(GraphBase, _len), 0, "Number of nodes"},
    {"_find_nodes_lookup_table",
     T_OBJECT_EX,
     offsetof(GraphBase, _find_nodes_lookup_table),
     0,
     "Lookup table for find_nodes"},
    {"_owning_module",
     T_OBJECT,
     offsetof(GraphBase, _owning_module),
     0,
     "The owning GraphModule"},
    {nullptr, 0, 0, 0, nullptr} // Sentinel
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyGetSetDef GraphBase_getset[] = {
    {"nodes",
     (getter)GraphBase_get_nodes,
     nullptr, // read-only
     (char*)"Get the list of Nodes in this Graph.",
     nullptr},
    {"owning_module",
     (getter)GraphBase_get_owning_module,
     (setter)GraphBase_set_owning_module,
     (char*)"The owning GraphModule, if any.",
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr} // Sentinel
};

static PyTypeObject GraphBaseType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._GraphBase", /* tp_name */
    sizeof(GraphBase), /* tp_basicsize */
    0, /* tp_itemsize */
    GraphBase_dealloc, /* tp_dealloc */
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
    "Base class for FX Graph, implemented in C++.", /* tp_doc */
    (traverseproc)GraphBase_traverse, /* tp_traverse */
    (inquiry)GraphBase_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    GraphBase_methods, /* tp_methods */
    GraphBase_members, /* tp_members */
    GraphBase_getset, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)GraphBase_init_fn, /* tp_init */
    nullptr, /* tp_alloc */
    GraphBase_new, /* tp_new */
};

} // namespace

////////////////////////////////
// C++ API for direct calls from node.cpp
////////////////////////////////

bool FindNodesLookupTable_contains_impl(
    PyObject* lookup_table,
    PyObject* node) {
  if (!lookup_table || lookup_table == Py_None) {
    return false;
  }
  if (!PyObject_TypeCheck(lookup_table, &FindNodesLookupTableType)) {
    return false;
  }
  int result = FindNodesLookupTable_contains(lookup_table, node);
  if (result < 0) {
    PyErr_Clear();
    return false;
  }
  return result != 0;
}

bool FindNodesLookupTable_remove_impl(PyObject* lookup_table, PyObject* node) {
  if (!lookup_table || lookup_table == Py_None) {
    return true;
  }
  if (!PyObject_TypeCheck(lookup_table, &FindNodesLookupTableType)) {
    return true;
  }
  PyObject* result = FindNodesLookupTable_remove(lookup_table, node);
  if (!result) {
    PyErr_Clear();
    return false;
  }
  Py_DECREF(result);
  return true;
}

bool FindNodesLookupTable_insert_impl(PyObject* lookup_table, PyObject* node) {
  if (!lookup_table || lookup_table == Py_None) {
    return true;
  }
  if (!PyObject_TypeCheck(lookup_table, &FindNodesLookupTableType)) {
    return true;
  }
  PyObject* result = FindNodesLookupTable_insert(lookup_table, node);
  if (!result) {
    PyErr_Clear();
    return false;
  }
  Py_DECREF(result);
  return true;
}

bool GraphBase_Check(PyObject* obj) {
  return obj && PyObject_TypeCheck(obj, &GraphBaseType);
}

PyObject* GraphBase_borrow_owning_module(PyObject* graph) {
  if (!GraphBase_Check(graph)) {
    return nullptr;
  }
  return reinterpret_cast<GraphBase*>(graph)->_owning_module;
}

PyObject* GraphBase_borrow_find_nodes_lookup_table(PyObject* graph) {
  if (!GraphBase_Check(graph)) {
    return nullptr;
  }
  return reinterpret_cast<GraphBase*>(graph)->_find_nodes_lookup_table;
}

bool Namespace_init(PyObject* module) {
  if (PyModule_AddType(module, &NamespaceBaseType) < 0) {
    return false;
  }
  return true;
}

bool FindNodesLookupTable_init(PyObject* module) {
  if (PyModule_AddType(module, &FindNodesLookupTableType) < 0) {
    return false;
  }
  return true;
}

bool NodeList_init(PyObject* module) {
  if (PyModule_AddType(module, &NodeListType) < 0) {
    return false;
  }
  return true;
}

bool GraphBase_init(PyObject* module) {
  if (PyModule_AddType(module, &GraphBaseType) < 0) {
    return false;
  }
  return true;
}

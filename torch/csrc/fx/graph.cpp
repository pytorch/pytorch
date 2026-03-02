#include <torch/csrc/fx/graph.h>

#include <structmember.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <string>

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

} // namespace

bool Namespace_init(PyObject* module) {
  if (PyModule_AddType(module, &NamespaceBaseType) < 0) {
    return false;
  }
  return true;
}

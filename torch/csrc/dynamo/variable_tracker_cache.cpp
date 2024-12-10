#include <torch/csrc/dynamo/variable_tracker_cache.h>
#include <torch/csrc/utils/python_ptr.h>

#include <functional>
#include <stdexcept>

namespace {

namespace pyptr = torch::pyptr;

struct VariableTrackerCacheKey {
  uintptr_t m_value;
  pyptr::Owned m_source;

  VariableTrackerCacheKey(pyptr::Borrowed value, pyptr::Borrowed source)
      : m_value(reinterpret_cast<uintptr_t>(value.ptr())),
        m_source(source.to_owned()) {}

  bool operator==(const VariableTrackerCacheKey& other) const {
    return (m_value == other.m_value) && m_source.equal(other.m_source);
  }

  size_t hash() const {
    return std::hash<uintptr_t>()(m_value) ^ m_source.hash();
  }
};

struct VariableTrackerCacheKeyHasher {
  std::size_t operator()(const VariableTrackerCacheKey& k) const {
    return k.hash();
  }
};

struct VariableTrackerCache {
  PyObject_HEAD

  std::unordered_map<
      VariableTrackerCacheKey,
      pyptr::Owned,
      VariableTrackerCacheKeyHasher>
      m_cache;

  ~VariableTrackerCache() {
    clear();
  }

  VariableTrackerCache() { // NOLINT(cppcoreguidelines-pro-type-member-init,
                           // modernize-use-equals-default)
    // don't use default constructor for objects w/ PyObject_HEAD (crashes).
    // CLANGTIDY complains about missing init for ob_base.
  }

  VariableTrackerCache(VariableTrackerCache&) = delete;
  VariableTrackerCache& operator=(VariableTrackerCache&) = delete;
  VariableTrackerCache(VariableTrackerCache&&) = delete;
  VariableTrackerCache& operator=(VariableTrackerCache&&) = delete;

  static PyObject* _new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    VariableTrackerCache* self =
        reinterpret_cast<VariableTrackerCache*>(type->tp_alloc(type, 0));
    if (self != nullptr) {
      new (self) VariableTrackerCache();
    }
    return reinterpret_cast<PyObject*>(self);
  }

  static void _dealloc(VariableTrackerCache* self) {
    self->~VariableTrackerCache();
    Py_TYPE(self)->tp_free((PyObject*)self);
  }

  void add(pyptr::Borrowed value, pyptr::Borrowed source, pyptr::Borrowed vt_) {
    auto vt = vt_.to_owned();
    auto key = VariableTrackerCacheKey(value, source);
    m_cache.emplace(std::move(key), vt);
  }

  void clear() {
    m_cache.clear();
  }

  pyptr::Owned clone() {
    throw std::runtime_error("unimplemented VariableTrackerCache::clone");
  }

  pyptr::Owned lookup(pyptr::Borrowed value, pyptr::Borrowed source) const {
    auto it = m_cache.find(VariableTrackerCacheKey(value, source));
    if (it == m_cache.end()) {
      return pyptr::Owned::none();
    } else {
      return it->second;
    }
  }
};

bool _checkParamCount(size_t nargs, size_t expected) {
  if (nargs < expected) {
    PyErr_SetString(PyExc_TypeError, "Too few parameters");
    return false;
  }
  if (nargs > expected) {
    PyErr_SetString(PyExc_TypeError, "Too many parameters");
    return false;
  }
  return true;
}

PyObject* _add(
    VariableTrackerCache* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (!_checkParamCount(nargs, 3))
    return Py_None;
  self->add(args[0], args[1], args[2]);
  return Py_None;
}

PyObject* _clear(
    VariableTrackerCache* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (!_checkParamCount(nargs, 0))
    return Py_None;
  self->clear();
  return Py_None;
}

PyObject* _clone(
    VariableTrackerCache* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (!_checkParamCount(nargs, 0))
    return Py_None;
  return self->clone().release();
}

PyObject* _lookup(
    VariableTrackerCache* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (!_checkParamCount(nargs, 2))
    return Py_None;
  return self->lookup(args[0], args[1]).release();
}

std::array<PyMethodDef, 5> vtc_methods = {
    PyMethodDef{
        "add",
        (PyCFunction)(void (*)()) & _add,
        METH_FASTCALL,
        nullptr},
    PyMethodDef{
        "clear",
        (PyCFunction)(void (*)()) & _clear,
        METH_FASTCALL,
        nullptr},
    PyMethodDef{
        "clone",
        (PyCFunction)(void (*)()) & _clone,
        METH_FASTCALL,
        nullptr},
    PyMethodDef{
        "lookup",
        (PyCFunction)(void (*)()) & _lookup,
        METH_FASTCALL,
        nullptr},
    PyMethodDef{nullptr, nullptr, 0, nullptr},
};

PyTypeObject vtc_type = {
  PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._dynamo.VariableTrackerCache", // tp_name
    sizeof(VariableTrackerCache), // tp_basicsize
    0, // tp_itemsize
    (destructor)VariableTrackerCache::_dealloc, // tp_dealloc
    0, // tp_print
    nullptr, // tp_getattr
    nullptr, // tp_setattr
    nullptr, // tp_reserved
    nullptr, // tp_repr
    nullptr, // tp_as_number
    nullptr, // tp_as_sequence
    nullptr, // tp_as_mapping
    nullptr, // tp_hash
    nullptr, // tp_call
    nullptr, // tp_str
    nullptr, // tp_getattro
    nullptr, // tp_setattro
    nullptr, // tp_as_buffer
    Py_TPFLAGS_DEFAULT, // tp_flags
    nullptr, // tp_doc
    nullptr, // tp_traverse
    nullptr, // tp_clear
    nullptr, // tp_richcompare
    0, // tp_weaklistoffset
    nullptr, // tp_iter
    nullptr, // tp_iternext
    vtc_methods.data(), // tp_methods
    nullptr, // tp_members
    nullptr, // tp_getset
    nullptr, // tp_base
    nullptr, // tp_dict
    nullptr, // tp_descr_get
    nullptr, // tp_descr_set
    0, // tp_dictoffset
    nullptr, // tp_init
    nullptr, // tp_alloc
    VariableTrackerCache::_new, // tp_new
};

} // anonymous namespace

void torch::dynamo::register_variable_tracker_cache(PyObject* mod) {
  if (PyType_Ready(&vtc_type) < 0) {
    return;
  }

  Py_INCREF(&vtc_type);
  if (PyModule_AddObject(mod, "VariableTrackerCache", (PyObject*)&vtc_type) <
      0) {
    Py_DECREF(&vtc_type);
    return;
  }
}

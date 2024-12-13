#include <torch/csrc/dynamo/variable_tracker_cache.h>
#include <torch/csrc/utils/object_ptr.h>

#include <array>
#include <functional>
#include <stdexcept>

namespace {

extern PyTypeObject vtc_type;

bool py_equal(const PyObject* a, const PyObject* b) {
  auto result = PyObject_RichCompareBool(
      const_cast<PyObject*>(a), // NOLINT(cppcoreguidelines-pro-type-const-cast)
      const_cast<PyObject*>(b), // NOLINT(cppcoreguidelines-pro-type-const-cast)
      Py_EQ);
  switch (result) {
    case 0:
      return false;
    case 1:
      return true;
    default:
      throw std::runtime_error("PyObject_RichCompareBool failed");
  }
}

Py_hash_t py_hash(const PyObject* p) {
  return PyObject_Hash(
      const_cast<PyObject*>( // NOLINT(cppcoreguidelines-pro-type-const-cast)
          p));
}

struct VariableTrackerCacheKey {
  /// We store the id(value) rather than the PyObject* so it's obvious that it
  /// shouldn't be reference counted or dereferenced.
  uintptr_t m_value;
  /// Two different source can point to the same object. However, Dynamo handles
  /// globals and local source differently when it comes to guards and possibly
  /// some other parts as well. So, cache also relies on the source.
  THPObjectPtr m_source;

  ~VariableTrackerCacheKey() = default;
  VariableTrackerCacheKey(PyObject* value, PyObject* source)
      : m_value(reinterpret_cast<uintptr_t>(value)),
        m_source(THPObjectPtr::dup(source)) {}
  VariableTrackerCacheKey(const VariableTrackerCacheKey& o)
      : m_value(o.m_value), m_source(o.m_source.dup()) {}
  VariableTrackerCacheKey& operator=(const VariableTrackerCacheKey&) = delete;
  VariableTrackerCacheKey(VariableTrackerCacheKey&& o) = default;
  VariableTrackerCacheKey& operator=(VariableTrackerCacheKey&& o) = delete;

  bool operator==(const VariableTrackerCacheKey& other) const {
    return (m_value == other.m_value) &&
        py_equal(m_source.get(), other.m_source.get());
  }

  size_t hash() const {
    return std::hash<uintptr_t>()(m_value) ^ py_hash(m_source.get());
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
      THPObjectPtr,
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

  void add(PyObject* value, PyObject* source, PyObject* vt_) {
    auto vt = THPObjectPtr::dup(vt_);
    VariableTrackerCacheKey key{value, source};
    m_cache.emplace(std::move(key), std::move(vt));
  }

  void clear() {
    m_cache.clear();
  }

  THPObjectPtr clone() {
    THPObjectPtr new_cache(PyObject_CallObject((PyObject*)&vtc_type, Py_None));
    VariableTrackerCache* p =
        reinterpret_cast<VariableTrackerCache*>(new_cache.get());
    for (const auto& i : m_cache) {
      p->m_cache.emplace(i.first, i.second.dup());
    }
    return new_cache;
  }

  THPObjectPtr lookup(PyObject* value, PyObject* source) const {
    auto it = m_cache.find(VariableTrackerCacheKey(value, source));
    if (it == m_cache.end()) {
      return THPObjectPtr::none();
    } else {
      return it->second.dup();
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
  if (_checkParamCount(nargs, 3)) {
    self->add(args[0], args[1], args[2]);
  }
  return THPObjectPtr::none().release();
}

PyObject* _clear(
    VariableTrackerCache* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (_checkParamCount(nargs, 0)) {
    self->clear();
  }
  return THPObjectPtr::none();
}

PyObject* _clone(
    VariableTrackerCache* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (!_checkParamCount(nargs, 0)) {
    return THPObjectPtr::none().release();
  }
  return self->clone().release();
}

PyObject* _lookup(
    VariableTrackerCache* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (!_checkParamCount(nargs, 2)) {
    return THPObjectPtr::none().release();
  }
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

const char* vtc_doc =
    "VariableTrackerCache is used to map (value, source) pairs to a corresponding "
    "VariableTracker.";

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
    vtc_doc, // tp_doc
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

#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/guards.h>

#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/dynamo/extra_state.h>

#include <atomic>

CacheEntry::CacheEntry(const py::handle& guarded_code, PyObject* backend)
    : backend{py::cast<py::object>(get_backend(backend))} {
  this->guard_manager = guarded_code.attr("guard_manager");
  this->code = guarded_code.attr("code");
  this->compile_id = guarded_code.attr("compile_id");
  py::object trace_annotation = guarded_code.attr("trace_annotation");
  const char* trace_annotation_str = PyUnicode_AsUTF8(trace_annotation.ptr());
  if (trace_annotation) {
    this->trace_annotation = std::string(trace_annotation_str);
  } else {
    this->trace_annotation = "Unknown";
  }
  this->root_mgr = torch::dynamo::convert_to_root_guard_manager(
      this->guard_manager.attr("root"));
  this->diff_guard_root_mgr = torch::dynamo::convert_to_root_guard_manager(
      this->guard_manager.attr("diff_guard_root"));
}

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED(
    "-Wdeprecated-copy-with-user-provided-dtor")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-copy-dtor")
// NOLINTNEXTLINE(bugprone-exception-escape)
CacheEntry::~CacheEntry() {
  // prevent guard_manager from use-after-free when invalidating
  this->guard_manager.attr("cache_entry") = py::none();
  this->guard_manager.attr("extra_state") = py::none();
}
C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()

CacheEntry::Detached CacheEntry::invalidate(py::object deleted_guard_manager) {
  Detached old{
      std::move(this->guard_manager),
      std::move(this->code),
      std::move(this->backend)};
  // The moved-from members are null, so these assignments decref nothing.
  this->guard_manager = std::move(deleted_guard_manager);
  this->code = py::none();
  this->backend = py::none();
  this->root_mgr = nullptr;
  this->diff_guard_root_mgr = nullptr;
  this->trace_annotation = "Invalidated";
  return old;
}

void CacheEntry::update_diff_guard_root_manager() {
  this->diff_guard_root_mgr = torch::dynamo::convert_to_root_guard_manager(
      this->guard_manager.attr("diff_guard_root"));
}

// Set once, the first time a backend that carries a cache key is built. Until
// then the lookup below is skipped: it is a MISS at every level of the callback
// chain for everyone who never precompiles, and py::hasattr on a const char*
// builds a str and raises-and-clears an AttributeError each time, on a path
// that runs per intercepted frame. Never cleared -- once such a backend exists
// in the process the chain can hold one for the rest of its life.
static std::atomic<bool> precompile_cache_keys_in_use{false};

void enable_precompile_cache_keys() {
  precompile_cache_keys_in_use.store(true, std::memory_order_relaxed);
}

// py::hasattr and PyObject_GetAttrString both build a str from the char* and
// then raise and clear an AttributeError on a miss, and every level of every
// intercepted frame's callback chain is a miss for the attributes below.
// Interned names plus the no-raise lookup keep the walk off the exception path
// entirely.
PyObject* lookup_optional_attr(py::handle obj, PyObject* name) {
  PyObject* value = nullptr;
#if IS_PYTHON_3_13_PLUS
  if (PyObject_GetOptionalAttr(obj.ptr(), name, &value) < 0) {
    PyErr_Clear();
    return nullptr;
  }
#else
  if (_PyObject_LookupAttr(obj.ptr(), name, &value) < 0) {
    PyErr_Clear();
    return nullptr;
  }
#endif
  // A borrowed reference is what the caller wants and what the chain keeps
  // alive; drop the one the lookup handed us rather than leak it.
  Py_XDECREF(value);
  return value;
}

// Returns a BORROWED reference, kept alive by the callback chain it was read
// off. Both attributes below must therefore be plain stored attributes, not
// properties or __getattr__ results, or the object dies with the temporary
// py::object this returns the pointer of.
PyObject* get_backend(PyObject* callback) {
  static PyObject* cache_key_name =
      PyUnicode_InternFromString("_torchdynamo_cache_key");
  static PyObject* orig_backend_name =
      PyUnicode_InternFromString("_torchdynamo_orig_backend");
  const bool check_cache_key =
      precompile_cache_keys_in_use.load(std::memory_order_relaxed);
  py::handle handle = py::handle(callback);
  while (true) {
    if (check_cache_key) {
      PyObject* key = lookup_optional_attr(handle, cache_key_name);
      if (key != nullptr) {
        return key;
      }
    }
    PyObject* next = lookup_optional_attr(handle, orig_backend_name);
    if (next == nullptr) {
      break;
    }
    handle = py::handle(next);
  }
  return handle.ptr();
}

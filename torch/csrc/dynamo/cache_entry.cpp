#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/guards.h>

#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/extra_state.h>

#include <atomic>

CacheEntry::CacheEntry(const py::handle& guarded_code, PyObject* backend)
    : backend{get_backend(backend)} {
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

void CacheEntry::invalidate(py::object deleted_guard_manager) {
  // Keep the current pointer alive but make the fields as if no-op
  this->guard_manager.attr("cache_entry") = py::none();
  this->guard_manager.attr("extra_state") = py::none();
  this->code = py::none();
  this->guard_manager = std::move(deleted_guard_manager);
  this->root_mgr = nullptr;
  this->diff_guard_root_mgr = nullptr;
  this->trace_annotation = "Invalidated";
  this->backend = py::none();
}

void CacheEntry::update_diff_guard_root_manager() {
  this->diff_guard_root_mgr = torch::dynamo::convert_to_root_guard_manager(
      this->guard_manager.attr("diff_guard_root"));
}

PyCodeObject* CacheEntry_get_code(CacheEntry* e) {
  return (PyCodeObject*)e->code.ptr();
}

const char* CacheEntry_get_trace_annotation(CacheEntry* e) {
  return e->trace_annotation.c_str();
}

PyObject* CacheEntry_to_obj(CacheEntry* e) {
  if (!e) {
    return py::none().release().ptr();
  }
  return py::cast(e, py::return_value_policy::reference).release().ptr();
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

// Owned on success, empty when absent, WITHOUT raising. py::hasattr and
// PyObject_GetAttrString both build a str from the char* and then raise and
// clear an AttributeError on a miss, and every level of every intercepted
// frame's callback chain is a miss for the attributes below. Interned names
// plus the no-raise lookup keep the walk off the exception path entirely.
py::object lookup_optional(py::handle handle, PyObject* name) {
  PyObject* value = nullptr;
  // pythoncapi_compat provides PyObject_GetOptionalAttr before 3.13.
  if (PyObject_GetOptionalAttr(handle.ptr(), name, &value) < 0) {
    PyErr_Clear();
    return py::object();
  }
  // Own the new reference (empty when the attribute is absent). A computed
  // attribute (property / __getattr__) has no other owner, so a borrowed
  // pointer would dangle the moment this returns.
  return py::reinterpret_steal<py::object>(value);
}

// Returns an OWNED reference, so the _torchdynamo_cache_key /
// _torchdynamo_orig_backend attributes may be computed (property /
// __getattr__) without the returned backend dangling.
py::object get_backend(PyObject* callback) {
  static PyObject* cache_key_name =
      PyUnicode_InternFromString("_torchdynamo_cache_key");
  static PyObject* orig_backend_name =
      PyUnicode_InternFromString("_torchdynamo_orig_backend");
  const bool check_cache_key =
      precompile_cache_keys_in_use.load(std::memory_order_relaxed);
  // `current` owns a reference at every step, so each attribute read runs
  // against a live object and the returned backend is never a dangling pointer
  // even when a link in the chain is a computed attribute.
  py::object current = py::reinterpret_borrow<py::object>(py::handle(callback));
  while (true) {
    if (check_cache_key) {
      py::object key = lookup_optional(current, cache_key_name);
      if (key) {
        return key;
      }
    }
    py::object next = lookup_optional(current, orig_backend_name);
    if (!next) {
      break;
    }
    current = std::move(next);
  }
  return current;
}

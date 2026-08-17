#include <atomic>

#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/guards.h>

#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/extra_state.h>

namespace {
std::atomic<uint64_t> next_cache_entry_identity{0};
}

CacheEntry::CacheEntry(const py::handle& guarded_code, PyObject* backend)
    : backend{py::cast<py::object>(get_backend(backend))},
      identity{++next_cache_entry_identity} {
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
  std::lock_guard<std::recursive_mutex> lock(this->state_mutex);
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
  std::lock_guard<std::recursive_mutex> lock(this->state_mutex);
  this->diff_guard_root_mgr = torch::dynamo::convert_to_root_guard_manager(
      this->guard_manager.attr("diff_guard_root"));
}

CacheEntryHandle::CacheEntryHandle(const std::shared_ptr<CacheEntry>& entry)
    : entry_(entry) {}

std::shared_ptr<CacheEntry> CacheEntryHandle::lock() const {
  return this->entry_.lock();
}

py::object CacheEntryHandle::backend() const {
  auto entry = this->lock();
  if (entry == nullptr) {
    return py::none();
  }
  std::lock_guard<std::recursive_mutex> lock(entry->state_mutex);
  return entry->backend;
}

void CacheEntryHandle::update_diff_guard_root_manager() const {
  auto entry = this->lock();
  if (entry != nullptr) {
    entry->update_diff_guard_root_manager();
  }
}

CacheEntrySnapshot::CacheEntrySnapshot(const CacheEntry& entry) {
  std::lock_guard<std::recursive_mutex> lock(entry.state_mutex);
  this->guard_manager = entry.guard_manager;
  this->code = entry.code;
  this->compile_id = entry.compile_id;
  this->backend = entry.backend;
  this->isolate_recompiles_id = entry._isolate_recompiles_id;
  this->trace_annotation = entry.trace_annotation;
  this->identity = entry.identity;
  this->root_mgr = entry.root_mgr;
  this->diff_guard_root_mgr = entry.diff_guard_root_mgr;
}

PyCodeObject* CacheEntry_get_code(CacheEntry* e) {
  return (PyCodeObject*)e->code.ptr();
}

const char* CacheEntry_get_trace_annotation(CacheEntry* e) {
  return e->trace_annotation.c_str();
}

// Returns a BORROWED reference, kept alive by the callback chain it was read
// off. Both attributes below must therefore be plain stored attributes, not
// properties or __getattr__ results, or the object dies with the temporary
// py::object this returns the pointer of.
PyObject* get_backend(PyObject* callback) {
  py::handle handle = py::handle(callback);
  while (true) {
    if (py::hasattr(handle, "_torchdynamo_cache_key")) {
      return handle.attr("_torchdynamo_cache_key").ptr();
    }
    if (!py::hasattr(handle, "_torchdynamo_orig_backend")) {
      break;
    }
    handle = handle.attr("_torchdynamo_orig_backend");
  }
  return handle.ptr();
}

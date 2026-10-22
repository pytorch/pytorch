#include <string>
#include <utility>

#include <c10/util/Exception.h>
#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/extra_state.h>
#include <torch/csrc/dynamo/guards.h>

CacheEntry::CacheEntry(const py::handle& guarded_code, PyObject* backend)
    : guard_manager{guarded_code.attr("guard_manager")},
      code{guarded_code.attr("code")},
      compile_id{guarded_code.attr("compile_id")},
      root_mgr{torch::dynamo::convert_to_root_guard_manager(
          guard_manager.attr("root"))},
      diff_guard_root_mgr{torch::dynamo::convert_to_root_guard_manager(
          guard_manager.attr("diff_guard_root"))},
      backend{py::cast<py::object>(get_backend(backend))} {
  if (py::object trace_annotation_obj{guarded_code.attr("trace_annotation")}) {
    trace_annotation = std::move(trace_annotation_obj).cast<std::string>();
  } else {
    trace_annotation = "Unknown";
  }
}

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED(
    "-Wdeprecated-copy-with-user-provided-dtor")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-copy-dtor")
// NOLINTNEXTLINE(bugprone-exception-escape)
CacheEntry::~CacheEntry() {
  // Only an entry linked into an ExtraState set the back-references.
  if (this->_owner == nullptr) {
    return;
  }
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
  TORCH_CHECK(
      this->_owner != nullptr,
      "cannot update the diff guard root of a copied cache entry");
  this->diff_guard_root_mgr = torch::dynamo::convert_to_root_guard_manager(
      this->guard_manager.attr("diff_guard_root"));
}

CacheEntry CacheEntry::copy_for_python() const {
  CacheEntry copy(*this);
  copy.root_mgr = nullptr;
  copy.diff_guard_root_mgr = nullptr;
  copy._owner = nullptr;
  copy._source = identity();
  copy._owner_loc = {};
  return copy;
}

const CacheEntry* CacheEntry::identity() const {
  return _source != nullptr ? _source : this;
}

bool CacheEntry::py_equal(const CacheEntry& other) const {
  return identity() == other.identity() && compile_id.is(other.compile_id);
}

std::size_t CacheEntry::py_hash() const {
  return std::hash<const void*>{}(identity());
}

PyObject* CacheEntry_to_obj(CacheEntry* e) {
  if (!e) {
    return py::none().release().ptr();
  }
  return py::cast(e, py::return_value_policy::reference).release().ptr();
}

static const py::str& get_orig_backend_str() {
  // NB: leak
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::str> storage;
  return storage
      .call_once_and_store_result([]() {
        return py::reinterpret_steal<py::str>(
            PyUnicode_InternFromString("_torchdynamo_orig_backend"));
      })
      .get_stored();
}

PyObject* get_backend(PyObject* callback) {
  py::handle handle{callback};
  const auto& orig_backend{get_orig_backend_str()};
  while (py::hasattr(handle, orig_backend)) {
    handle = handle.attr(orig_backend);
  }
  return handle.ptr();
}

#include <atomic>
#include <string>
#include <utility>

#include <c10/util/Exception.h>
#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/extra_state.h>
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

namespace {
std::atomic<uint64_t> next_serial{0};
} // namespace

CacheEntry::CacheEntry(const py::handle& guarded_code, PyObject* backend)
    : guard_manager{guarded_code.attr("guard_manager")},
      code{guarded_code.attr("code")},
      compile_id{guarded_code.attr("compile_id")},
      root_mgr{torch::dynamo::convert_to_root_guard_manager(
          guard_manager.attr("root"))},
      diff_guard_root{guard_manager.attr("diff_guard_root")},
      diff_guard_root_mgr{
          torch::dynamo::convert_to_root_guard_manager(diff_guard_root)},
      backend{py::cast<py::object>(get_backend(backend))},
      serial{next_serial.fetch_add(1, std::memory_order_relaxed)} {
  if (py::object trace_annotation_obj{guarded_code.attr("trace_annotation")}) {
    trace_annotation = std::move(trace_annotation_obj).cast<std::string>();
  } else {
    trace_annotation = "Unknown";
  }
}

void CacheEntry::invalidate(
    py::object deleted_guard_manager,
    std::vector<py::object>& released) {
  // Keep the current pointer alive but make the fields as if no-op.
  released.push_back(
      std::exchange(this->guard_manager, std::move(deleted_guard_manager)));
  released.push_back(std::exchange(this->code, py::none()));
  released.push_back(std::exchange(this->diff_guard_root, py::none()));
  released.push_back(std::exchange(this->backend, py::none()));
  this->root_mgr = nullptr;
  this->diff_guard_root_mgr = nullptr;
  this->trace_annotation = "Invalidated";
}

CacheEntry CacheEntry::copy_for_python() const {
  CacheEntry copy(*this);
  copy.root_mgr = nullptr;
  copy.diff_guard_root_mgr = nullptr;
  copy._owner = nullptr;
  copy._owner_loc = {};
  return copy;
}

bool CacheEntry::py_equal(const CacheEntry& other) const {
  return serial == other.serial;
}

std::size_t CacheEntry::py_hash() const {
  return std::hash<uint64_t>{}(serial);
}

static PyObject* get_orig_backend_str() {
  // NB: leak
  constinit static py::gil_safe_call_once_and_store<py::str> storage;
  return storage
      .call_once_and_store_result([]() {
        return py::reinterpret_steal<py::str>(
            PyUnicode_InternFromString("_torchdynamo_orig_backend"));
      })
      .get_stored()
      .ptr();
}

PyObject* get_backend(PyObject* callback) {
  PyObject* orig_backend_str{get_orig_backend_str()};
  PyObject* next{nullptr};

  // PyObject_GetOptionalAttr is in the C-API compat file, and more efficient
  // than calling hasattr/getattr.
  while (PyObject_GetOptionalAttr(callback, orig_backend_str, &next) > 0) {
    // Return borrowed; the previous hop's attribute keeps it alive.
    Py_DECREF(next);
    callback = next;
  }
  return callback;
}

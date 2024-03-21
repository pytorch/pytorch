#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/guards.h>

#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/extra_state.h>

CacheEntry::CacheEntry(const py::handle& guarded_code, PyObject* backend) {
  this->check_fn = guarded_code.attr("check_fn");
  this->code = guarded_code.attr("code");
  this->backend = backend;
  // TODO - clean this up when enable_cpp_guard_manager is True by default
  if (py::hasattr(this->check_fn, "root")) {
    this->root_mgr = convert_to_root_guard_manager(this->check_fn.attr("root"));
  }
}

CacheEntry::~CacheEntry() {
  // prevent check_fn from use-after-free when invalidating
  this->check_fn.attr("cache_entry") = py::none();
  this->check_fn.attr("extra_state") = py::none();
}

py::object CacheEntry::next() {
  NULL_CHECK(this->_owner);
  auto it = this->_owner_loc;
  ++it;
  if (it == this->_owner->cache_entry_list.end()) {
    return py::none();
  }
  return py::cast(*it, py::return_value_policy::reference);
}

PyCodeObject* CacheEntry_get_code(CacheEntry* e) {
  return (PyCodeObject*)e->code.ptr();
}

PyObject* CacheEntry_to_obj(CacheEntry* e) {
  if (!e) {
    return py::none().release().ptr();
  }
  return py::cast(e, py::return_value_policy::reference).release().ptr();
}

PyObject* get_backend(PyObject* callback) {
  py::handle handle = py::handle(callback);
  while (py::hasattr(handle, "_torchdynamo_orig_callable")) {
    handle = handle.attr("_torchdynamo_orig_callable");
  }
  if (py::hasattr(handle, "compiler_fn")) {
    handle = handle.attr("compiler_fn");
  }
  return handle.ptr();
}

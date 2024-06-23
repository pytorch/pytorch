#include <torch/csrc/dynamo/extra_state.h>

#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/utils/python_compat.h>

#if IS_PYTHON_3_12_PLUS
#define _PyCode_GetExtra PyUnstable_Code_GetExtra
#define _PyCode_SetExtra PyUnstable_Code_SetExtra
#endif

Py_ssize_t extra_index = -1;

CacheEntry* ExtraState::get_first_entry() {
  if (this->cache_entry_list.empty()) {
    return nullptr;
  }
  return &this->cache_entry_list.front();
}

void ExtraState::move_to_front(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);
  CHECK(!this->cache_entry_list.empty());
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  this->cache_entry_list.splice(
      this->cache_entry_list.begin(),
      this->cache_entry_list,
      cache_entry->_owner_loc);
}

void ExtraState::invalidate(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);
  CHECK(!this->cache_entry_list.empty());
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  this->cache_entry_list.erase(cache_entry->_owner_loc);
}

CacheEntry* extract_cache_entry(ExtraState* extra_state) {
  if (extra_state == nullptr || extra_state == SKIP_CODE) {
    return nullptr;
  }
  return extra_state->get_first_entry();
}

FrameState* extract_frame_state(ExtraState* extra_state) {
  if (extra_state == nullptr || extra_state == SKIP_CODE) {
    return nullptr;
  }
  return (FrameState*)extra_state->frame_state.ptr();
}

ExtraState* get_extra_state(PyCodeObject* code) {
  ExtraState* extra = nullptr;
  _PyCode_GetExtra((PyObject*)code, extra_index, (void**)&extra);
  return extra;
}

void destroy_extra_state(void* obj) {
  ExtraState* extra = (ExtraState*)obj;
  if (extra != nullptr && extra != SKIP_CODE) {
    delete extra;
  }
}

void set_extra_state(PyCodeObject* code, ExtraState* extra_state) {
  ExtraState* old_extra_state = get_extra_state(code);
  CHECK(
      old_extra_state == nullptr || old_extra_state == SKIP_CODE ||
      old_extra_state != extra_state);
  _PyCode_SetExtra((PyObject*)code, extra_index, extra_state);
}

ExtraState* init_and_set_extra_state(PyCodeObject* code) {
  // Invariant - Extra state should not have been set before, therefore it
  // should be nullptr.
  CHECK(get_extra_state(code) == nullptr);
  ExtraState* extra_state = new ExtraState();
  NULL_CHECK(extra_state);
  set_extra_state(code, extra_state);
  return extra_state;
}

PyObject* lookup(
    ExtraState* extra_state,
    PyObject* f_locals,
    const PyObject* backend) {
  size_t index = 0;
  CacheEntry* found = nullptr;
  py::handle locals(f_locals);
  for (CacheEntry& cache_entry : extra_state->cache_entry_list) {
    // Check backend. Py_False means run only mode.
    bool valid = backend == Py_False || cache_entry.backend == backend;
    if (valid) {
      try {
        // TODO(anijain2305) - Clean this up when enable_cpp_guard_manager is
        // True by default
        if (cache_entry.root_mgr != nullptr) {
          valid = torch::dynamo::run_root_guard_manager(
              cache_entry.root_mgr, f_locals);
        } else {
          valid = cache_entry.check_fn(locals).cast<bool>();
        }
      } catch (py::error_already_set& e) {
        if (guard_error_hook) {
          py::handle guard_error_hook_handle(guard_error_hook);
          guard_error_hook_handle(
              cache_entry.check_fn,
              cache_entry.code,
              locals,
              index,
              index == extra_state->cache_entry_list.size() - 1);
        }
        // this function is called from C, so we cannot repropagate
        // the exception
        e.restore();
        return nullptr;
      }
    }
    if (valid) {
      found = &cache_entry;
      break;
    }
    ++index;
  }
  if (found) {
    extra_state->move_to_front(found);
    return found->code.ptr();
  }
  return py::none().ptr();
}

CacheEntry* create_cache_entry(
    ExtraState* extra_state,
    PyObject* guarded_code,
    PyObject* backend) {
  extra_state->cache_entry_list.emplace_front(guarded_code, backend);
  auto new_iter = extra_state->cache_entry_list.begin();
  new_iter->_owner = extra_state;
  new_iter->_owner_loc = new_iter;
  // Set check_fn references to extra_state and CacheEntry
  // Warning: lifetime is controlled by C++!
  py::handle check_fn = py::handle(guarded_code).attr("check_fn");
  check_fn.attr("cache_entry") =
      py::cast(*new_iter, py::return_value_policy::reference);
  check_fn.attr("extra_state") =
      py::cast(extra_state, py::return_value_policy::reference);
  return &*new_iter;
}

py::list _debug_get_cache_entry_list(const py::handle& code_obj) {
  if (!py::isinstance(code_obj, py::module::import("types").attr("CodeType"))) {
    throw py::type_error("expected a code object!");
  }
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  py::list result;
  if (extra && extra != SKIP_CODE) {
    for (CacheEntry& e : extra->cache_entry_list) {
      result.append(py::cast(e, py::return_value_policy::reference));
    }
  }
  return result;
}

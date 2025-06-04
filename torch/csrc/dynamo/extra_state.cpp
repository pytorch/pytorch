#include <torch/csrc/dynamo/extra_state.h>

#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>
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

ExtraState::ExtraState(PyCodeObject* orig_code_arg)
    : orig_code(orig_code_arg) {}

void ExtraState::move_to_front(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);
  CHECK(!this->cache_entry_list.empty());
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  this->cache_entry_list.splice(
      this->cache_entry_list.begin(),
      this->cache_entry_list,
      cache_entry->_owner_loc);
}

void ExtraState::move_to_back(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);
  CHECK(!this->cache_entry_list.empty());
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  this->cache_entry_list.splice(
      this->cache_entry_list.end(),
      this->cache_entry_list,
      cache_entry->_owner_loc);
}

void ExtraState::invalidate(
    CacheEntry* cache_entry,
    py::object deleted_guard_manager) {
  // Sometimes setting the cache_entry->code to None causes the orig_code to be
  // freed. This calls destroy_extra_state, which deletes the extra_state and
  // all the cache_entries. This causes the `this` pointer to be a dangling
  // pointer, causing a segfault. So, we manually inc/dec ref the original code
  // pointer to prevent triggering of destroy_extra_state while the invalidate
  // function is running.
  Py_INCREF(this->orig_code);

  CHECK(cache_entry->_owner == this);
  CHECK(!this->cache_entry_list.empty());
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  cache_entry->invalidate(std::move(deleted_guard_manager));
  // Move the cache entry to the end of the list because these will always
  // return False.
  cache_entry->_owner->move_to_back(cache_entry);
  Py_DECREF(this->orig_code);
}

CacheEntry* extract_cache_entry(ExtraState* extra_state) {
  if (extra_state == nullptr) {
    return nullptr;
  }
  return extra_state->get_first_entry();
}

FrameState* extract_frame_state(ExtraState* extra_state) {
  if (extra_state == nullptr) {
    return nullptr;
  }
  return (FrameState*)extra_state->frame_state.ptr();
}

FrameExecStrategy extra_state_get_exec_strategy(ExtraState* extra_state) {
  return extra_state->strategy;
}

void extra_state_set_exec_strategy(
    ExtraState* extra_state,
    FrameExecStrategy strategy) {
  extra_state->strategy = strategy;
}

ExtraState* get_extra_state(PyCodeObject* code) {
  ExtraState* extra = nullptr;
  _PyCode_GetExtra((PyObject*)code, extra_index, (void**)&extra);
  return extra;
}

void destroy_extra_state(void* obj) {
  ExtraState* extra = (ExtraState*)obj;
  delete extra;
}

void set_extra_state(PyCodeObject* code, ExtraState* extra_state) {
  ExtraState* old_extra_state = get_extra_state(code);
  CHECK(extra_state == nullptr || old_extra_state != extra_state);
  _PyCode_SetExtra((PyObject*)code, extra_index, extra_state);
}

ExtraState* init_and_set_extra_state(PyCodeObject* code) {
  // Invariant - Extra state should not have been set before, therefore it
  // should be nullptr.
  CHECK(get_extra_state(code) == nullptr);
  ExtraState* extra_state = new ExtraState(code);
  NULL_CHECK(extra_state);
  set_extra_state(code, extra_state);
  // freed by destroy_extra_state (since we need to pass these objects to C)
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return extra_state;
}

static bool backend_match(PyObject* saved_backend, PyObject* backend) {
  // Pointer equality check for common case
  if (saved_backend != backend) {
    // The Py_TYPE check should not be required but there is a pre-existing
    // issue where backend is possibly deallocated (or nullptr) and causes
    // segfaults. Check test - test_inplace_custom_op_intermediate
    return (
        Py_TYPE(saved_backend) == Py_TYPE(backend) &&
        PyObject_RichCompareBool(saved_backend, backend, Py_EQ));
  }
  return true;
}

void lookup(
    ExtraState* extra_state,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    PyObject** maybe_cached_code,
    const char** trace_annotation,
    bool is_skip_guard_eval_unsafe) {
  size_t index = 0;
  CacheEntry* found = nullptr;
  for (CacheEntry& cache_entry : extra_state->cache_entry_list) {
    // Check backend. Py_False means run only mode.

    bool valid =
        backend == Py_False || backend_match(cache_entry.backend, backend);

    if (valid) {
      try {
        if (is_skip_guard_eval_unsafe) {
          valid = torch::dynamo::run_root_guard_manager(
              cache_entry.diff_guard_root_mgr, f_locals);
        } else {
          valid = torch::dynamo::run_root_guard_manager(
              cache_entry.root_mgr, f_locals);
        }
      } catch (py::error_already_set& e) {
        if (guard_error_hook) {
          py::handle guard_error_hook_handle(guard_error_hook);
          py::handle f_locals_dict = (PyObject*)f_locals->to_dict();
          guard_error_hook_handle(
              cache_entry.guard_manager,
              cache_entry.code,
              f_locals_dict,
              index,
              index == extra_state->cache_entry_list.size() - 1);
        }
        // this function is called from C, so we cannot repropagate
        // the exception
        e.restore();
        *maybe_cached_code = nullptr;
        return;
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
    *maybe_cached_code = found->code.ptr();
    *trace_annotation = found->trace_annotation.c_str();
    return;
  }
  *maybe_cached_code = py::none().ptr();
}

CacheEntry* create_cache_entry(
    ExtraState* extra_state,
    PyObject* guarded_code,
    PyObject* backend) {
  extra_state->cache_entry_list.emplace_front(guarded_code, backend);
  auto new_iter = extra_state->cache_entry_list.begin();
  new_iter->_owner = extra_state;
  new_iter->_owner_loc = new_iter;
  // Set guard_manager references to extra_state and CacheEntry
  // Warning: lifetime is controlled by C++!
  py::handle guard_manager = py::handle(guarded_code).attr("guard_manager");
  guard_manager.attr("cache_entry") =
      py::cast(*new_iter, py::return_value_policy::reference);
  guard_manager.attr("extra_state") =
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
  if (extra != nullptr) {
    for (CacheEntry& e : extra->cache_entry_list) {
      result.append(py::cast(e, py::return_value_policy::reference));
    }
  }
  return result;
}

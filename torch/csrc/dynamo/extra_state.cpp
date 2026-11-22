#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include <c10/util/Exception.h>
#include <torch/csrc/dynamo/extra_state.h>

#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>
#include <torch/csrc/dynamo/guards.h>

#if IS_PYTHON_3_12_PLUS
#define _PyCode_GetExtra PyUnstable_Code_GetExtra
#define _PyCode_SetExtra PyUnstable_Code_SetExtra
#endif

namespace {
// Short-term fix for: https://github.com/pytorch/pytorch/issues/166926
#ifdef Py_GIL_DISABLED
// Off by default on free-threaded builds: reordering on every hit contends on
// the code object's lock.  It is safe to turn on with _set_lru_cache(true).
bool use_lru = false;
#else
bool use_lru = true;
#endif

// Everything that reaches into an ExtraState runs under a critical section on
// the owning code object.  The code object outlives the ExtraState, so locking
// on it also stops reset_code from freeing the state while another thread is
// inside one of these calls.
//
// The section is released whenever the thread detaches: a GC pause, a
// contended lock, or any Python code at all, guard evaluation included.  So the
// bodies below only copy and relink; anything that can run Python happens
// outside, on references taken inside.
//
// CodeLock is a scoped Py_BEGIN_CRITICAL_SECTION, so a C++ exception still
// releases it.  It compiles away on GIL builds.
class CodeLock {
 public:
  explicit CodeLock([[maybe_unused]] PyCodeObject* code) {
#ifdef Py_GIL_DISABLED
    PyCriticalSection_Begin(&section_, (PyObject*)code);
#endif
  }
  ~CodeLock() {
#ifdef Py_GIL_DISABLED
    PyCriticalSection_End(&section_);
#endif
  }
  CodeLock(const CodeLock&) = delete;
  CodeLock& operator=(const CodeLock&) = delete;

 private:
#ifdef Py_GIL_DISABLED
  PyCriticalSection section_;
#endif
};

// Set by set_extra_state while it overwrites the slot, so destroy_extra_state
// hands the old state back instead of deleting it there.
thread_local ExtraState** deferred_destroy = nullptr;

ExtraState* get_extra_state(PyCodeObject* code) {
  ExtraState* extra = nullptr;
  _PyCode_GetExtra((PyObject*)code, extra_index, (void**)&extra);
  return extra;
}

ExtraState* get_or_init_extra_state(PyCodeObject* code) {
  ExtraState* extra_state = get_extra_state(code);
  if (extra_state != nullptr) {
    return extra_state;
  }
  extra_state = new ExtraState();
  NULL_CHECK(extra_state);
  _PyCode_SetExtra((PyObject*)code, extra_index, extra_state);
  // freed by destroy_extra_state (since we need to pass these objects to C)
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return extra_state;
}

// The region's own bucket, then the default bucket, matching lookup order.
template <typename Fn>
void for_each_bucket(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id,
    Fn&& fn) {
  int64_t ids_to_search[] = {isolate_recompiles_id, -1};
  int num_ids = (isolate_recompiles_id >= 0) ? 2 : 1;
  for (int i = 0; i < num_ids; i++) {
    auto it = extra_state->cache_entry_map.find(ids_to_search[i]);
    if (it != extra_state->cache_entry_map.end()) {
      fn(ids_to_search[i], it->second);
    }
  }
}

// The live entry a candidate was copied from, or null if it has since been
// reset away.
CacheEntry* find_entry(
    ExtraState* extra_state,
    const CacheCandidate& candidate) {
  auto it = extra_state->cache_entry_map.find(candidate.isolate_recompiles_id);
  if (it == extra_state->cache_entry_map.end()) {
    return nullptr;
  }
  for (CacheEntry& entry : it->second) {
    if (entry.serial == candidate.serial) {
      return &entry;
    }
  }
  return nullptr;
}

FrameExecStrategy get_region_exec_strategy(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id) {
  if (isolate_recompiles_id < 0) {
    return extra_state->strategy;
  }
  FrameExecStrategy result{DEFAULT, DEFAULT};
  auto it = extra_state->region_strategy_map.find(isolate_recompiles_id);
  if (it != extra_state->region_strategy_map.end()) {
    result = it->second;
  }
  // Isolated regions inherit SKIP from the global strategy (deliberate
  // "do not trace" marks from skip_code / @torch._dynamo.skip / FX
  // plumbing / TorchScript __init__ / etc.) but do NOT inherit
  // RUN_ONLY, which can only come from a prior non-isolated
  // recompile-limit hit and would otherwise poison every new region.
  FrameExecStrategy global = extra_state->strategy;
  if (global.cur_action == FrameAction::SKIP) {
    result.cur_action = FrameAction::SKIP;
  }
  if (global.recursive_action == FrameAction::SKIP) {
    result.recursive_action = FrameAction::SKIP;
  }
  return result;
}

void set_region_exec_strategy(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id,
    FrameExecStrategy strategy) {
  if (isolate_recompiles_id < 0) {
    extra_state->strategy = strategy;
  } else {
    extra_state->region_strategy_map[isolate_recompiles_id] = strategy;
  }
}

bool backend_match(PyObject* saved_backend, PyObject* backend) {
  // Pointer equality check for common case
  if (saved_backend != backend) {
    int result = PyObject_RichCompareBool(saved_backend, backend, Py_EQ);
    // Check for exception
    if (result == -1) {
      PyErr_Clear();
      return false;
    }
    return (result == 1);
  }
  return true;
}

bool candidate_backend_matches(
    const CacheCandidate& candidate,
    PyObject* backend) {
  return Py_IsFalse(backend) || candidate.backend_is_frame_backend ||
      backend_match(candidate.backend.ptr(), backend);
}

bool candidate_has_no_guards(
    const CacheCandidate& candidate,
    bool is_skip_guard_eval_unsafe) {
  if (is_skip_guard_eval_unsafe && candidate.diff_guard_root_mgr != nullptr) {
    return torch::dynamo::root_guard_manager_has_no_guards(
        candidate.diff_guard_root_mgr);
  }
  return torch::dynamo::root_guard_manager_has_no_guards(candidate.root_mgr);
}

bool run_candidate_guards(
    const CacheCandidate& candidate,
    FrameLocalsMapping* f_locals,
    bool is_skip_guard_eval_unsafe) {
  if (is_skip_guard_eval_unsafe) {
    return candidate_has_no_guards(
               candidate, /*is_skip_guard_eval_unsafe=*/true) ||
        torch::dynamo::run_root_guard_manager(
               candidate.diff_guard_root_mgr, f_locals);
  }
  return torch::dynamo::run_root_guard_manager(candidate.root_mgr, f_locals);
}

void report_guard_error(
    const CacheCandidate& candidate,
    FrameLocalsMapping* f_locals) {
  if (guard_error_hook == nullptr) {
    return;
  }
  py::handle guard_error_hook_handle(guard_error_hook);
  py::handle f_locals_dict = (PyObject*)f_locals->to_dict();
  guard_error_hook_handle(
      candidate.guard_manager,
      candidate.code,
      f_locals_dict,
      candidate.index,
      candidate.index == candidate.bucket_size - 1);
}

// Re-enters the critical section to take the hit's code and move it to the
// front.  Returns false if the candidate's entry is gone or no longer holds
// code.
bool commit_hit(
    PyCodeObject* code,
    const CacheCandidate& candidate,
    CacheLookupResult* result) {
  CodeLock lock(code);
  result->code = py::none();
  ExtraState* extra_state = get_extra_state(code);
  if (extra_state == nullptr) {
    return false;
  }
  CacheEntry* entry = find_entry(extra_state, candidate);
  if (entry == nullptr || !PyCode_Check(entry->code.ptr())) {
    return false;
  }
  if (use_lru) {
    extra_state->move_to_front(
        entry, extra_state->cache_entry_list(candidate.isolate_recompiles_id));
  }
  result->code = entry->code;
  result->trace_annotation = entry->trace_annotation;
  return true;
}

// The code object whose cache holds guard_manager, from the weak reference
// record_compile_result leaves on it, or None.
py::object cached_code(const py::handle& guard_manager) {
  py::object ref = py::getattr(guard_manager, "code_ref", py::none());
  if (ref.is_none()) {
    return ref;
  }
  py::object code = ref();
  return PyCode_Check(code.ptr()) ? code : py::none();
}

// The live entries holding guard_manager in the code object's cache, none if
// they have been invalidated or reset away.  Called under the code object's
// lock.  A guard manager can be shared by entries if a callback reuses it.
std::vector<CacheEntry*> find_entries(
    PyCodeObject* code,
    PyObject* guard_manager) {
  std::vector<CacheEntry*> found;
  ExtraState* extra_state = get_extra_state(code);
  if (extra_state == nullptr) {
    return found;
  }
  for (auto& bucket : extra_state->cache_entry_map) {
    for (CacheEntry& entry : bucket.second) {
      if (entry.guard_manager.ptr() == guard_manager) {
        found.push_back(&entry);
      }
    }
  }
  return found;
}

} // namespace

Py_ssize_t extra_index = -1;

std::list<CacheEntry>& ExtraState::cache_entry_list(
    int64_t isolate_recompiles_id) {
  return this->cache_entry_map[isolate_recompiles_id];
}

bool ExtraState::has_any_cache_entries() const {
  return this->total_cache_entry_count > 0;
}

void ExtraState::move_to_front(
    CacheEntry* cache_entry,
    std::list<CacheEntry>& entries) {
  CHECK(cache_entry->_owner == this);
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  entries.splice(entries.begin(), entries, cache_entry->_owner_loc);
}

void ExtraState::move_to_back(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  auto& list = this->cache_entry_map[cache_entry->_isolate_recompiles_id];
  list.splice(list.end(), list, cache_entry->_owner_loc);
}

bool _invalidate_cache_entry(
    const py::handle& guard_manager,
    py::object deleted_guard_manager) {
  // Owning the code object keeps its ExtraState alive until released, declared
  // before the lock, drops the entry's old references after it.
  py::object code = cached_code(guard_manager);
  if (code.is_none()) {
    return false;
  }
  std::vector<py::object> released;
  CodeLock lock((PyCodeObject*)code.ptr());
  std::vector<CacheEntry*> entries =
      find_entries((PyCodeObject*)code.ptr(), guard_manager.ptr());
  for (CacheEntry* entry : entries) {
    entry->invalidate(deleted_guard_manager, released);
    // Move the cache entry to the end of the list because these will always
    // return False.
    entry->_owner->move_to_back(entry);
  }
  return !entries.empty();
}

void _update_diff_guard_root_manager(const py::handle& guard_manager) {
  py::object code = cached_code(guard_manager);
  if (code.is_none()) {
    return;
  }
  py::object root = guard_manager.attr("diff_guard_root");
  void* root_mgr = torch::dynamo::convert_to_root_guard_manager(root);
  // Declared before the lock, so the old roots are dropped unlocked.
  std::vector<py::object> released;
  CodeLock lock((PyCodeObject*)code.ptr());
  for (CacheEntry* entry :
       find_entries((PyCodeObject*)code.ptr(), guard_manager.ptr())) {
    released.push_back(std::exchange(entry->diff_guard_root, root));
    entry->diff_guard_root_mgr = root_mgr;
  }
}

void destroy_extra_state(void* obj) {
  ExtraState* extra = (ExtraState*)obj;
  if (deferred_destroy != nullptr) {
    *deferred_destroy = extra;
    return;
  }
  delete extra;
}

void set_extra_state(PyCodeObject* code, ExtraState* extra_state) {
  // Python 3.12 frees the old value before it stores the new one, and freeing
  // cache entries runs Python, so another thread could otherwise find a state
  // being torn down still in the slot.  Clear the slot first, then delete.
  ExtraState* old_extra_state = nullptr;
  {
    CodeLock lock(code);
    CHECK(extra_state == nullptr || get_extra_state(code) != extra_state);
    deferred_destroy = &old_extra_state;
    _PyCode_SetExtra((PyObject*)code, extra_index, extra_state);
    deferred_destroy = nullptr;
  }
  delete old_extra_state;
}

bool get_frame_exec_strategy(
    PyCodeObject* code,
    int64_t isolate_recompiles_id,
    bool create,
    FrameExecStrategy* strategy) {
  CodeLock lock(code);
  ExtraState* extra_state =
      create ? get_or_init_extra_state(code) : get_extra_state(code);
  if (extra_state == nullptr) {
    return false;
  }
  *strategy = get_region_exec_strategy(extra_state, isolate_recompiles_id);
  return true;
}

void set_code_exec_strategy(PyCodeObject* code, FrameExecStrategy strategy) {
  CodeLock lock(code);
  get_or_init_extra_state(code)->strategy = strategy;
}

CacheSnapshot snapshot_cache(
    PyCodeObject* code,
    int64_t isolate_recompiles_id,
    PyObject* backend,
    bool is_skip_guard_eval_unsafe) {
  CacheSnapshot snapshot;
  CodeLock lock(code);
  ExtraState* extra_state = get_extra_state(code);
  if (extra_state == nullptr) {
    return snapshot;
  }
  for (const PrecompileEntry& entry : extra_state->precompile_entries) {
    snapshot.precompile_entries.push_back(
        PrecompileCandidate{entry.guard_manager, entry.code, entry.root_mgr});
  }
  for_each_bucket(
      extra_state,
      isolate_recompiles_id,
      [&](int64_t id, std::list<CacheEntry>& entries) {
        snapshot.has_relevant_entries = true;
        size_t index = 0;
        for (const CacheEntry& entry : entries) {
          CacheCandidate& candidate = snapshot.entries.emplace_back();
          candidate.guard_manager = entry.guard_manager;
          candidate.root_mgr = entry.root_mgr;
          if (is_skip_guard_eval_unsafe) {
            candidate.diff_guard_root = entry.diff_guard_root;
            candidate.diff_guard_root_mgr = entry.diff_guard_root_mgr;
          }
          candidate.backend_is_frame_backend =
              Py_IsFalse(backend) || entry.backend.ptr() == backend;
          if (!candidate.backend_is_frame_backend) {
            candidate.backend = entry.backend;
          }
          candidate.code = entry.code;
          candidate.isolate_recompiles_id = id;
          candidate.serial = entry.serial;
          candidate.index = index++;
          candidate.bucket_size = entries.size();
        }
      });
  return snapshot;
}

bool try_lookup_without_guard_eval(
    PyCodeObject* code,
    const CacheSnapshot& snapshot,
    PyObject* backend,
    bool is_skip_guard_eval_unsafe,
    CacheLookupResult* result) {
  if (!snapshot.precompile_entries.empty()) {
    // Only the first precompile entry can be safely fast-pathed: a later
    // guardless entry must not preempt an earlier guarded entry whose guards
    // may pass.
    const auto& entry = snapshot.precompile_entries.front();
    if (torch::dynamo::root_guard_manager_has_no_guards(entry.root_mgr)) {
      result->code = entry.code;
      return true;
    }
    return false;
  }

  for (const CacheCandidate& candidate : snapshot.entries) {
    if (!candidate_backend_matches(candidate, backend)) {
      continue;
    }
    if (!PyCode_Check(candidate.code.ptr())) {
      continue;
    }
    if (!candidate_has_no_guards(candidate, is_skip_guard_eval_unsafe)) {
      return false;
    }
    if (commit_hit(code, candidate, result)) {
      return true;
    }
  }

  result->code = py::none();
  return true;
}

void lookup(
    PyCodeObject* code,
    const CacheSnapshot& snapshot,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    bool is_skip_guard_eval_unsafe,
    CacheLookupResult* result) {
  for (const auto& entry : snapshot.precompile_entries) {
    if (torch::dynamo::run_root_guard_manager(entry.root_mgr, f_locals)) {
      result->code = entry.code;
      return;
    }
  }

  for (const CacheCandidate& candidate : snapshot.entries) {
    if (!candidate_backend_matches(candidate, backend)) {
      continue;
    }
    bool valid = false;
    try {
      valid =
          run_candidate_guards(candidate, f_locals, is_skip_guard_eval_unsafe);
    } catch (py::error_already_set& e) {
      report_guard_error(candidate, f_locals);
      e.restore();
      result->code = py::object();
      return;
    }
    if (valid && commit_hit(code, candidate, result)) {
      return;
    }
  }
  result->code = py::none();
}

CompileInputs get_compile_inputs(
    PyCodeObject* code,
    int64_t isolate_recompiles_id) {
  CompileInputs inputs;
  // Wrapped for Python after the lock, as allocating can run Python.
  std::optional<CacheEntry> cache_entry;
  {
    CodeLock lock(code);
    ExtraState* extra_state = get_or_init_extra_state(code);
    inputs.frame_state = extra_state->frame_state;
    // Search own bucket first, then fall back to default bucket (-1),
    // matching lookup() behavior.
    for_each_bucket(
        extra_state,
        isolate_recompiles_id,
        [&](int64_t, std::list<CacheEntry>& entries) {
          if (!cache_entry && !entries.empty()) {
            cache_entry = entries.front().copy_for_python();
          }
        });
  }
  if (cache_entry) {
    inputs.cache_entry =
        py::cast(std::move(*cache_entry), py::return_value_policy::move);
  }
  return inputs;
}

CacheLookupResult record_compile_result(
    PyCodeObject* code,
    int64_t isolate_recompiles_id,
    bool apply_to_code,
    FrameExecStrategy new_strategy,
    PyObject* guarded_code,
    PyObject* backend) {
  CacheLookupResult result{py::none(), ""};
  // Built outside the critical section, since reading guarded_code and
  // setting attributes run Python, then spliced in, which does not.
  std::list<CacheEntry> pending;
  if (!Py_IsNone(guarded_code)) {
    CacheEntry& entry = pending.emplace_back(guarded_code, backend);
    entry.guard_manager.attr("code_ref") = py::weakref((PyObject*)code);
    result.code = entry.code;
    result.trace_annotation = entry.trace_annotation;
  }

  int64_t id = get_current_isolate_recompiles_id();
  CodeLock lock(code);
  ExtraState* extra_state = get_or_init_extra_state(code);
  if (apply_to_code) {
    set_region_exec_strategy(extra_state, isolate_recompiles_id, new_strategy);
  }
  if (!pending.empty()) {
    auto& entries = extra_state->cache_entry_list(id);
    auto new_iter = pending.begin();
    entries.splice(
        use_lru ? entries.begin() : entries.end(), pending, new_iter);
    new_iter->_owner = extra_state;
    new_iter->_owner_loc = new_iter;
    new_iter->_isolate_recompiles_id = id;
    extra_state->total_cache_entry_count++;
  }
  return result;
}

static PyCodeObject* code_from_handle(const py::handle& code_obj) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  return (PyCodeObject*)code_obj.ptr();
}

// Callers copy entries out before building the list, as allocating Python
// objects can run GC finalizers that mutate the cache being walked.
template <typename T>
static py::list to_py_list(std::vector<T>&& entries) {
  py::list result;
  for (T& e : entries) {
    result.append(py::cast(std::move(e), py::return_value_policy::move));
  }
  return result;
}

py::list _debug_get_cache_entry_list(const py::handle& code_obj) {
  PyCodeObject* code = code_from_handle(code_obj);
  std::vector<CacheEntry> copies;
  {
    CodeLock lock(code);
    ExtraState* extra = get_extra_state(code);
    if (extra != nullptr) {
      // Sort by isolate_recompiles_id for deterministic iteration order.
      std::vector<int64_t> ids;
      ids.reserve(extra->cache_entry_map.size());
      for (auto& kv : extra->cache_entry_map) {
        ids.push_back(kv.first);
      }
      std::sort(ids.begin(), ids.end());
      for (int64_t id : ids) {
        for (const CacheEntry& e : extra->cache_entry_map[id]) {
          copies.push_back(e.copy_for_python());
        }
      }
    }
  }
  return to_py_list(std::move(copies));
}

py::list _get_cache_entries_for_region(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id) {
  TORCH_CHECK(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  std::vector<CacheEntry> copies;
  {
    CodeLock lock(code);
    ExtraState* extra = get_extra_state(code);
    if (extra != nullptr) {
      auto it = extra->cache_entry_map.find(isolate_recompiles_id);
      if (it != extra->cache_entry_map.end()) {
        for (const CacheEntry& e : it->second) {
          copies.push_back(e.copy_for_python());
        }
      }
    }
  }
  return to_py_list(std::move(copies));
}

size_t _get_total_cache_entry_count(const py::handle& code_obj) {
  TORCH_CHECK(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  CodeLock lock(code);
  ExtraState* extra = get_extra_state(code);
  if (extra == nullptr) {
    return 0;
  }
  return extra->total_cache_entry_count;
}

PrecompileEntry::PrecompileEntry(py::object gm, py::object c)
    : guard_manager(std::move(gm)), code(std::move(c)) {
  TORCH_CHECK(
      PyCode_Check(code.ptr()), "Expecting CodeType from PrecompileEntry.");
  root_mgr =
      torch::dynamo::convert_to_root_guard_manager(guard_manager.attr("root"));
}

void _reset_precompile_entries(const py::handle& code_obj) {
  PyCodeObject* code = code_from_handle(code_obj);
  // Declared before the lock, so they are dropped after it is released, as
  // dropping them runs Python.
  std::list<PrecompileEntry> removed;
  CodeLock lock(code);
  ExtraState* extra = get_extra_state(code);
  if (extra != nullptr) {
    removed.swap(extra->precompile_entries);
  }
}

void _load_precompile_entry(
    const py::handle& code_obj,
    py::object guard_manager,
    py::object dynamo_code) {
  PyCodeObject* code = code_from_handle(code_obj);
  auto entry =
      PrecompileEntry(std::move(guard_manager), std::move(dynamo_code));
  CodeLock lock(code);
  get_or_init_extra_state(code)->precompile_entries.push_back(std::move(entry));
}

bool _set_lru_cache(py::object boolean) {
  bool prior = use_lru;
  use_lru = py::cast<bool>(boolean);
  return prior;
}

py::list _debug_get_precompile_entries(const py::handle& code_obj) {
  PyCodeObject* code = code_from_handle(code_obj);
  std::vector<PrecompileEntry> copies;
  {
    CodeLock lock(code);
    ExtraState* extra = get_extra_state(code);
    if (extra != nullptr) {
      copies.assign(
          extra->precompile_entries.begin(), extra->precompile_entries.end());
    }
  }
  return to_py_list(std::move(copies));
}

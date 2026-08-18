#include <algorithm>
#include <mutex>
#include <vector>

#include <c10/util/Exception.h>
#include <torch/csrc/dynamo/extra_state.h>

#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/utils/python_compat.h>

#if IS_PYTHON_3_12_PLUS
#define _PyCode_GetExtra PyUnstable_Code_GetExtra
#define _PyCode_SetExtra PyUnstable_Code_SetExtra
#endif

namespace {
// Short-term fix for: https://github.com/pytorch/pytorch/issues/166926
bool use_lru = true;

// Strategy tokens come from a process-wide counter so that resetting one code
// object's ExtraState cannot make a stale owner's token look current again.
// Only strategy WRITES touch this, so its mutex is off the per-frame read path,
// which locks the ExtraState's own strategy_mutex instead.
uint64_t next_strategy_generation = 0;
std::mutex generation_mutex;
std::mutex extra_state_init_mutex;

uint64_t next_generation() {
  std::lock_guard<std::mutex> lock(generation_mutex);
  return ++next_strategy_generation;
}

// Acquiring a mutex while holding the GIL deadlocks against a thread that holds
// the mutex and needs the GIL, and lookup() does exactly that: guard evaluation
// runs Python (LAMBDA_GUARD calls back into the interpreter), which can drop
// the GIL at any bytecode boundary. Take the uncontended fast path without
// touching the GIL, and release it before blocking so the owner can finish.
template <typename Mutex>
class PythonAwareLock {
 public:
  explicit PythonAwareLock(Mutex& mutex) : lock_(mutex, std::try_to_lock) {
    if (lock_.owns_lock()) {
      return;
    }
    if (PyGILState_Check()) {
      py::gil_scoped_release release;
      lock_.lock();
    } else {
      lock_.lock();
    }
  }

  PythonAwareLock(const PythonAwareLock&) = delete;
  PythonAwareLock& operator=(const PythonAwareLock&) = delete;
  PythonAwareLock(PythonAwareLock&&) = delete;
  PythonAwareLock& operator=(PythonAwareLock&&) = delete;
  ~PythonAwareLock() = default;

 private:
  std::unique_lock<Mutex> lock_;
};

using CacheLock = PythonAwareLock<std::recursive_mutex>;
using ExtraStateInitLock = PythonAwareLock<std::mutex>;
} // namespace

Py_ssize_t extra_index = -1;

ExtraState::ExtraState(PyCodeObject* orig_code_arg)
    : orig_code(orig_code_arg) {}

CacheEntryList& ExtraState::cache_entry_list(int64_t isolate_recompiles_id) {
  return this->cache_entry_map[isolate_recompiles_id];
}

bool ExtraState::has_any_cache_entries() const {
  CacheLock lock(this->cache_mutex);
  return this->total_cache_entry_count > 0;
}

bool ExtraState::has_relevant_entries(int64_t isolate_recompiles_id) const {
  CacheLock lock(this->cache_mutex);
  return std::any_of(
             this->precompile_entries.begin(),
             this->precompile_entries.end(),
             [isolate_recompiles_id](const PrecompileEntryPtr& entry) {
               return entry->isolate_recompiles_id == isolate_recompiles_id;
             }) ||
      this->cache_entry_map.count(isolate_recompiles_id) > 0 ||
      (isolate_recompiles_id >= 0 && this->cache_entry_map.count(-1) > 0);
}

void ExtraState::move_to_front(
    CacheEntry* cache_entry,
    CacheEntryList& entries) {
  CHECK(cache_entry->_owner == this);
  CHECK(cache_entry == cache_entry->_owner_loc->get());
  entries.splice(entries.begin(), entries, cache_entry->_owner_loc);
}

void ExtraState::move_to_back(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);
  CHECK(cache_entry == cache_entry->_owner_loc->get());
  auto& list = this->cache_entry_map[cache_entry->_isolate_recompiles_id];
  list.splice(list.end(), list, cache_entry->_owner_loc);
}

void ExtraState::clear_cache_entries(
    std::vector<CacheEntryPtr>& retired_cache_entries) {
  for (auto& [id, entries] : this->cache_entry_map) {
    (void)id;
    for (CacheEntryPtr& entry : entries) {
      entry->_owner = nullptr;
      retired_cache_entries.push_back(std::move(entry));
    }
  }
  this->cache_entry_map.clear();
  this->total_cache_entry_count = 0;
}

void ExtraState::clear_cache_entries_for_region(
    int64_t isolate_recompiles_id,
    std::vector<CacheEntryPtr>& retired_cache_entries) {
  auto it = this->cache_entry_map.find(isolate_recompiles_id);
  if (it == this->cache_entry_map.end()) {
    return;
  }
  TORCH_CHECK(this->total_cache_entry_count >= it->second.size());
  this->total_cache_entry_count -= it->second.size();
  for (CacheEntryPtr& entry : it->second) {
    entry->_owner = nullptr;
    retired_cache_entries.push_back(std::move(entry));
  }
  this->cache_entry_map.erase(it);
}

void ExtraState::clear_precompile_entries(
    std::vector<PrecompileEntryPtr>& retired_precompile_entries) {
  for (PrecompileEntryPtr& entry : this->precompile_entries) {
    retired_precompile_entries.push_back(std::move(entry));
  }
  this->precompile_entries.clear();
}

void ExtraState::clear_precompile_entries_for_region(
    int64_t isolate_recompiles_id,
    std::vector<PrecompileEntryPtr>& retired_precompile_entries) {
  for (auto it = this->precompile_entries.begin();
       it != this->precompile_entries.end();) {
    if ((*it)->isolate_recompiles_id == isolate_recompiles_id) {
      retired_precompile_entries.push_back(std::move(*it));
      it = this->precompile_entries.erase(it);
    } else {
      ++it;
    }
  }
}

void ExtraState::reset() {
  std::vector<CacheEntryPtr> retired_cache_entries;
  std::vector<PrecompileEntryPtr> retired_precompile_entries;
  py::object retired_frame_state;
  {
    CacheLock cache_lock(this->cache_mutex);
    PythonAwareLock<std::mutex> frame_state_lock(this->frame_state_mutex);
    PythonAwareLock<std::mutex> strategy_lock(this->strategy_mutex);
    ++this->cache_generation;
    this->clear_cache_entries(retired_cache_entries);
    this->clear_precompile_entries(retired_precompile_entries);
    retired_frame_state = std::move(this->frame_state);
    this->frame_state = py::dict();
    this->strategy = FrameExecStrategy{DEFAULT, DEFAULT};
    this->strategy_generation = 0;
    this->region_strategy_map.clear();
  }
}

void ExtraState::invalidate(
    const CacheEntryHandle& cache_entry,
    py::object deleted_guard_manager,
    py::object live_guard_manager) {
  // Sometimes setting the cache_entry->code to None causes the orig_code to be
  // freed. This calls destroy_extra_state, which deletes the extra_state and
  // all the cache_entries. This causes the `this` pointer to be a dangling
  // pointer, causing a segfault. So, we manually inc/dec ref the original code
  // pointer to prevent triggering of destroy_extra_state while the invalidate
  // function is running.
  Py_INCREF(this->orig_code);
  {
    CacheLock lock(this->cache_mutex);
    CacheEntryPtr expected_entry = cache_entry.lock();
    CacheEntryPtr live_entry;
    for (auto& [id, entries] : this->cache_entry_map) {
      (void)id;
      for (CacheEntryPtr& live : entries) {
        std::lock_guard<std::recursive_mutex> state_lock(live->state_mutex);
        if (live->guard_manager.ptr() == live_guard_manager.ptr()) {
          live_entry = live;
          break;
        }
      }
      if (live_entry) {
        break;
      }
    }
    if (live_entry) {
      CHECK(live_entry == expected_entry);
      CHECK(live_entry->_owner == this);
      CHECK(live_entry.get() == live_entry->_owner_loc->get());
      live_entry->invalidate(std::move(deleted_guard_manager));
      // Move the cache entry to the end of the list because these will always
      // return False.
      if (live_entry->_owner == this) {
        this->move_to_back(live_entry.get());
      }
    }
  }
  // The lock must be released BEFORE the decref: if this drops the last
  // reference, destroy_extra_state deletes `this` along with cache_mutex, and
  // unlocking a destroyed mutex is undefined behaviour.
  Py_DECREF(this->orig_code);
}

py::object extract_cache_entry_snapshot(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id) {
  if (extra_state == nullptr) {
    return py::none();
  }
  CacheLock lock(extra_state->cache_mutex);
  // Search own bucket first, then fall back to default bucket (-1),
  // matching lookup() behavior.
  int64_t ids_to_search[] = {isolate_recompiles_id, -1};
  int num_ids = (isolate_recompiles_id >= 0) ? 2 : 1;

  for (int i = 0; i < num_ids; i++) {
    auto it = extra_state->cache_entry_map.find(ids_to_search[i]);
    if (it != extra_state->cache_entry_map.end() && !it->second.empty()) {
      return py::cast(CacheEntrySnapshot(*it->second.front()));
    }
  }
  return py::none();
}

FrameState* extract_frame_state(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id) {
  (void)isolate_recompiles_id;
  if (extra_state == nullptr) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(extra_state->frame_state_mutex);
  PyObject* frame_state = extra_state->frame_state.ptr();
  Py_INCREF(frame_state);
  return frame_state;
}

FrameExecStrategy extra_state_get_exec_strategy(ExtraState* extra_state) {
  std::lock_guard<std::mutex> lock(extra_state->strategy_mutex);
  return extra_state->strategy;
}

uint64_t extra_state_get_exec_strategy_token(
    ExtraState* extra_state,
    FrameExecStrategy* strategy) {
  std::lock_guard<std::mutex> lock(extra_state->strategy_mutex);
  *strategy = extra_state->strategy;
  return extra_state->strategy_generation;
}

static void set_exec_strategy_unlocked(
    ExtraState* extra_state,
    FrameExecStrategy strategy) {
  extra_state->strategy = strategy;
  extra_state->strategy_generation = next_generation();
}

void extra_state_set_exec_strategy(
    ExtraState* extra_state,
    FrameExecStrategy strategy) {
  std::lock_guard<std::mutex> lock(extra_state->strategy_mutex);
  set_exec_strategy_unlocked(extra_state, strategy);
}

uint64_t extra_state_set_exec_strategy_with_token(
    ExtraState* extra_state,
    FrameExecStrategy strategy,
    FrameExecStrategy* prior_strategy) {
  std::lock_guard<std::mutex> lock(extra_state->strategy_mutex);
  *prior_strategy = extra_state->strategy;
  set_exec_strategy_unlocked(extra_state, strategy);
  return extra_state->strategy_generation;
}

bool extra_state_compare_and_set_exec_strategy(
    ExtraState* extra_state,
    uint64_t expected_generation,
    FrameExecStrategy strategy) {
  std::lock_guard<std::mutex> lock(extra_state->strategy_mutex);
  if (extra_state->strategy_generation != expected_generation) {
    return false;
  }
  set_exec_strategy_unlocked(extra_state, strategy);
  return true;
}

FrameExecStrategy extra_state_get_region_exec_strategy(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id) {
  std::lock_guard<std::mutex> lock(extra_state->strategy_mutex);
  if (isolate_recompiles_id < 0) {
    return extra_state->strategy;
  }
  auto it = extra_state->region_strategy_map.find(isolate_recompiles_id);
  if (it != extra_state->region_strategy_map.end()) {
    return it->second;
  }
  // Isolated regions inherit SKIP from the global strategy (deliberate
  // "do not trace" marks from skip_code / @torch._dynamo.skip / FX
  // plumbing / TorchScript __init__ / etc.) but do NOT inherit
  // RUN_ONLY, which can only come from a prior non-isolated
  // recompile-limit hit and would otherwise poison every new region.
  FrameExecStrategy global = extra_state->strategy;
  FrameExecStrategy result{DEFAULT, DEFAULT};
  if (global.cur_action == FrameAction::SKIP) {
    result.cur_action = FrameAction::SKIP;
  }
  if (global.recursive_action == FrameAction::SKIP) {
    result.recursive_action = FrameAction::SKIP;
  }
  return result;
}

void extra_state_set_region_exec_strategy(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id,
    FrameExecStrategy strategy) {
  if (isolate_recompiles_id < 0) {
    extra_state_set_exec_strategy(extra_state, strategy);
  } else {
    std::lock_guard<std::mutex> lock(extra_state->strategy_mutex);
    extra_state->region_strategy_map[isolate_recompiles_id] = strategy;
  }
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

void reset_extra_state(PyCodeObject* code) {
  ExtraState* extra_state = get_extra_state(code);
  if (extra_state != nullptr) {
    extra_state->reset();
  }
}

ExtraState* init_and_set_extra_state(PyCodeObject* code) {
  ExtraState* extra_state = get_extra_state(code);
  if (extra_state != nullptr) {
    return extra_state;
  }
  ExtraStateInitLock lock(extra_state_init_mutex);
  extra_state = get_extra_state(code);
  if (extra_state != nullptr) {
    return extra_state;
  }
  extra_state = new ExtraState(code);
  NULL_CHECK(extra_state);
  set_extra_state(code, extra_state);
  // freed by destroy_extra_state (since we need to pass these objects to C)
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return extra_state;
}

static bool backend_match(PyObject* saved_backend, PyObject* backend) {
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

static bool cache_entry_has_no_guards(
    const CacheEntrySnapshot& cache_entry,
    bool is_skip_guard_eval_unsafe) {
  if (is_skip_guard_eval_unsafe && cache_entry.diff_guard_root_mgr != nullptr) {
    return torch::dynamo::root_guard_manager_has_no_guards(
        cache_entry.diff_guard_root_mgr);
  }
  return torch::dynamo::root_guard_manager_has_no_guards(cache_entry.root_mgr);
}

struct CacheEntryCandidate {
  CacheEntryPtr entry;
  CacheEntrySnapshot snapshot;
  int64_t bucket_id;
  size_t index;
  size_t entry_count;
};

struct LookupSnapshot {
  uint64_t generation;
  std::vector<PrecompileEntryPtr> precompile_entries;
  std::vector<CacheEntryCandidate> cache_entries;
};

static bool cache_entry_is_current(const CacheEntryCandidate& candidate) {
  CacheEntrySnapshot current(*candidate.entry);
  return current.identity == candidate.snapshot.identity &&
      current.state_generation == candidate.snapshot.state_generation &&
      current.code.ptr() == candidate.snapshot.code.ptr() &&
      current.guard_manager.ptr() == candidate.snapshot.guard_manager.ptr();
}

static LookupSnapshot snapshot_lookup(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id) {
  CacheLock lock(extra_state->cache_mutex);
  LookupSnapshot result{extra_state->cache_generation, {}, {}};
  for (const PrecompileEntryPtr& entry : extra_state->precompile_entries) {
    if (entry->isolate_recompiles_id == isolate_recompiles_id) {
      result.precompile_entries.push_back(entry);
    }
  }

  int64_t ids_to_search[] = {isolate_recompiles_id, -1};
  int num_ids = (isolate_recompiles_id >= 0) ? 2 : 1;
  for (int i = 0; i < num_ids; ++i) {
    auto it = extra_state->cache_entry_map.find(ids_to_search[i]);
    if (it == extra_state->cache_entry_map.end()) {
      continue;
    }
    size_t index = 0;
    size_t entry_count = it->second.size();
    for (const CacheEntryPtr& entry : it->second) {
      result.cache_entries.push_back(CacheEntryCandidate{
          entry,
          CacheEntrySnapshot(*entry),
          ids_to_search[i],
          index++,
          entry_count});
    }
  }
  return result;
}

static bool finish_cache_hit(
    ExtraState* extra_state,
    uint64_t generation,
    const CacheEntryCandidate& candidate,
    CacheLookupResult* result) {
  CacheLock lock(extra_state->cache_mutex);
  if (extra_state->cache_generation != generation ||
      candidate.entry->_owner != extra_state ||
      candidate.entry->_isolate_recompiles_id != candidate.bucket_id ||
      !cache_entry_is_current(candidate)) {
    return false;
  }
  auto it = extra_state->cache_entry_map.find(candidate.bucket_id);
  if (it == extra_state->cache_entry_map.end()) {
    return false;
  }
  if (use_lru) {
    extra_state->move_to_front(candidate.entry.get(), it->second);
  }
  result->code = candidate.snapshot.code;
  result->trace_annotation = candidate.snapshot.trace_annotation;
  return true;
}

static bool finish_precompile_hit(
    ExtraState* extra_state,
    uint64_t generation,
    const PrecompileEntryPtr& entry,
    CacheLookupResult* result) {
  CacheLock lock(extra_state->cache_mutex);
  if (extra_state->cache_generation != generation) {
    return false;
  }
  result->code = entry->code;
  return true;
}

static bool lookup_generation_is_current(
    ExtraState* extra_state,
    uint64_t generation) {
  CacheLock lock(extra_state->cache_mutex);
  return extra_state->cache_generation == generation;
}

static bool run_cache_entry_guard(
    const CacheEntryCandidate& candidate,
    FrameLocalsMapping* f_locals,
    bool is_skip_guard_eval_unsafe) {
  if (is_skip_guard_eval_unsafe) {
    return cache_entry_has_no_guards(
               candidate.snapshot, /*is_skip_guard_eval_unsafe=*/true) ||
        torch::dynamo::run_root_guard_manager(
               candidate.snapshot.diff_guard_root_mgr, f_locals);
  }
  return torch::dynamo::run_root_guard_manager(
      candidate.snapshot.root_mgr, f_locals);
}

static void run_guard_error_hook(
    const CacheEntryCandidate& candidate,
    FrameLocalsMapping* f_locals) {
  if (!guard_error_hook) {
    return;
  }
  py::handle guard_error_hook_handle(guard_error_hook);
  py::handle f_locals_dict = (PyObject*)f_locals->to_dict();
  guard_error_hook_handle(
      candidate.snapshot.guard_manager,
      candidate.snapshot.code,
      f_locals_dict,
      candidate.index,
      candidate.index == candidate.entry_count - 1);
}

static bool backend_matches(
    const CacheEntryCandidate& candidate,
    PyObject* backend) {
  return Py_IsFalse(backend) ||
      backend_match(candidate.snapshot.backend.ptr(), backend);
}

static size_t next_bucket_index(
    const std::vector<CacheEntryCandidate>& candidates,
    size_t start) {
  int64_t bucket_id = candidates[start].bucket_id;
  while (start < candidates.size() &&
         candidates[start].bucket_id == bucket_id) {
    ++start;
  }
  return start;
}

static bool try_lookup_bucket_without_guard_eval(
    ExtraState* extra_state,
    const LookupSnapshot& snapshot,
    size_t begin,
    size_t end,
    PyObject* backend,
    bool is_skip_guard_eval_unsafe,
    CacheLookupResult* result) {
  for (size_t i = begin; i < end; ++i) {
    const CacheEntryCandidate& candidate = snapshot.cache_entries[i];
    if (!backend_matches(candidate, backend)) {
      continue;
    }
    if (!PyCode_Check(candidate.snapshot.code.ptr())) {
      continue;
    }
    if (!cache_entry_has_no_guards(
            candidate.snapshot, is_skip_guard_eval_unsafe) ||
        !cache_entry_is_current(candidate)) {
      return false;
    }
    return finish_cache_hit(
        extra_state, snapshot.generation, candidate, result);
  }
  return true;
}

void lookup(
    ExtraState* extra_state,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    CacheLookupResult* result,
    bool is_skip_guard_eval_unsafe) {
  result->code = py::object();
  result->trace_annotation.clear();
  LookupSnapshot snapshot =
      snapshot_lookup(extra_state, isolate_recompiles_id);

  // Precompile entries match their OWN region only, deliberately unlike the
  // cache-entry fallback below. The identity guards that would tell two
  // artifacts of one model apart are exactly the ones precompile has to drop,
  // so a fallback here serves another artifact's graph for a call this region
  // does not cover, instead of the miss that serving() turns into a loud error.
  // The cache-entry fallback is narrower than it looks but is not a precedent:
  // lookup_in_list also requires backend_match, though note that short-circuits
  // when the backend is Py_False, which is every frame under run-only. Callers
  // that install for an isolated region must pass its id (see
  // CompilePackage.install) rather than rely on the default bucket.
  for (const PrecompileEntryPtr& entry : snapshot.precompile_entries) {
    if (torch::dynamo::run_root_guard_manager(entry->root_mgr, f_locals)) {
      if (!finish_precompile_hit(
              extra_state, snapshot.generation, entry, result)) {
        result->code = py::none();
      }
      return;
    }
  }

  // Search own bucket first, then fall back to default bucket (-1).
  // This lets isolated compiles reuse compilations from non-isolated
  // torch.compile() calls (BC friendly). New entries are still written
  // to the isolated bucket.
  for (const CacheEntryCandidate& candidate : snapshot.cache_entries) {
    if (!backend_matches(candidate, backend)) {
      continue;
    }
    bool valid = false;
    try {
      valid = run_cache_entry_guard(
          candidate, f_locals, is_skip_guard_eval_unsafe);
    } catch (py::error_already_set& e) {
      run_guard_error_hook(candidate, f_locals);
      e.restore();
      return;
    }
    if (valid && finish_cache_hit(
                     extra_state, snapshot.generation, candidate, result)) {
      return;
    }
    if (!lookup_generation_is_current(extra_state, snapshot.generation)) {
      result->code = py::none();
      return;
    }
    if (!cache_entry_is_current(candidate)) {
      result->code = py::none();
      return;
    }
  }
  result->code = py::none();
}

bool try_lookup_without_guard_eval(
    ExtraState* extra_state,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    CacheLookupResult* result,
    bool is_skip_guard_eval_unsafe) {
  result->code = py::object();
  result->trace_annotation.clear();
  LookupSnapshot snapshot =
      snapshot_lookup(extra_state, isolate_recompiles_id);
  if (!snapshot.precompile_entries.empty()) {
    // Only the first precompile entry can be safely fast-pathed: a later
    // guardless entry must not preempt an earlier guarded entry whose guards
    // may pass.
    const PrecompileEntryPtr& first = snapshot.precompile_entries.front();
    if (!torch::dynamo::root_guard_manager_has_no_guards(first->root_mgr)) {
      return false;
    }
    return finish_precompile_hit(
        extra_state, snapshot.generation, first, result);
  }

  size_t begin = 0;
  while (begin < snapshot.cache_entries.size()) {
    size_t end = next_bucket_index(snapshot.cache_entries, begin);
    if (!try_lookup_bucket_without_guard_eval(
            extra_state,
            snapshot,
            begin,
            end,
            backend,
            is_skip_guard_eval_unsafe,
            result)) {
      return false;
    }
    if (result->code) {
      return true;
    }
    begin = end;
  }

  if (!lookup_generation_is_current(extra_state, snapshot.generation)) {
    return false;
  }
  result->code = py::none();
  return true;
}

CacheEntrySnapshot create_cache_entry(
    ExtraState* extra_state,
    PyObject* guarded_code,
    PyObject* backend) {
  CacheLock lock(extra_state->cache_mutex);
  int64_t id = get_current_isolate_recompiles_id();
  auto& entries = extra_state->cache_entry_list(id);
  CacheEntryPtr new_entry =
      std::make_shared<CacheEntry>(py::handle(guarded_code), backend);
  CacheEntryList::iterator new_iter;
  if (use_lru) {
    entries.emplace_front(new_entry);
    new_iter = entries.begin();
  } else {
    entries.emplace_back(new_entry);
    new_iter = std::prev(entries.end());
  }
  new_entry->_owner = extra_state;
  new_entry->_owner_loc = new_iter;
  new_entry->_isolate_recompiles_id = id;
  extra_state->total_cache_entry_count++;
  py::handle guard_manager = py::handle(guarded_code).attr("guard_manager");
  guard_manager.attr("cache_entry") = py::cast(CacheEntryHandle(new_entry));
  guard_manager.attr("extra_state") =
      py::cast(extra_state, py::return_value_policy::reference);
  return CacheEntrySnapshot(*new_entry);
}

py::list _debug_get_cache_entry_list(const py::handle& code_obj) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  py::list result;
  if (extra != nullptr) {
    CacheLock lock(extra->cache_mutex);
    // Sort by isolate_recompiles_id for deterministic iteration order.
    std::vector<int64_t> ids;
    ids.reserve(extra->cache_entry_map.size());
    for (auto& kv : extra->cache_entry_map) {
      ids.push_back(kv.first);
    }
    std::sort(ids.begin(), ids.end());
    for (int64_t id : ids) {
      for (const CacheEntryPtr& entry : extra->cache_entry_map[id]) {
        result.append(py::cast(CacheEntrySnapshot(*entry)));
      }
    }
  }
  return result;
}

py::list _get_cache_entries_for_region(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id) {
  TORCH_CHECK(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  py::list result;
  if (extra != nullptr) {
    CacheLock lock(extra->cache_mutex);
    auto it = extra->cache_entry_map.find(isolate_recompiles_id);
    if (it != extra->cache_entry_map.end()) {
      for (const CacheEntryPtr& entry : it->second) {
        result.append(py::cast(CacheEntrySnapshot(*entry)));
      }
    }
  }
  return result;
}

void _clear_cache_entries_for_region(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  TORCH_CHECK_VALUE(
      isolate_recompiles_id >= 0, "cannot clear the default cache region");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  if (extra == nullptr) {
    return;
  }
  std::vector<CacheEntryPtr> retired_cache_entries;
  {
    CacheLock cache_lock(extra->cache_mutex);
    PythonAwareLock<std::mutex> strategy_lock(extra->strategy_mutex);
    ++extra->cache_generation;
    extra->clear_cache_entries_for_region(
        isolate_recompiles_id, retired_cache_entries);
    extra->region_strategy_map.erase(isolate_recompiles_id);
  }
}

size_t _get_total_cache_entry_count(const py::handle& code_obj) {
  TORCH_CHECK(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  if (extra == nullptr) {
    return 0;
  }
  CacheLock lock(extra->cache_mutex);
  return extra->total_cache_entry_count;
}

PrecompileEntry::PrecompileEntry(py::object gm, py::object c, int64_t region_id)
    : guard_manager(std::move(gm)),
      code(std::move(c)),
      isolate_recompiles_id(region_id) {
  TORCH_CHECK(
      PyCode_Check(code.ptr()), "Expecting CodeType from PrecompileEntry.");
  root_manager = guard_manager.attr("root");
  root_mgr = torch::dynamo::convert_to_root_guard_manager(root_manager);
}

PrecompileEntrySnapshot::PrecompileEntrySnapshot(const PrecompileEntry& entry)
    : guard_manager(entry.guard_manager),
      isolate_recompiles_id(entry.isolate_recompiles_id) {}

void _reset_precompile_entries(const py::handle& code_obj) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  if (extra != nullptr) {
    std::vector<PrecompileEntryPtr> retired_precompile_entries;
    {
      CacheLock lock(extra->cache_mutex);
      ++extra->cache_generation;
      extra->clear_precompile_entries(retired_precompile_entries);
    }
  }
}

void _reset_precompile_entries_for_region(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  if (extra != nullptr) {
    std::vector<PrecompileEntryPtr> retired_precompile_entries;
    {
      CacheLock lock(extra->cache_mutex);
      ++extra->cache_generation;
      extra->clear_precompile_entries_for_region(
          isolate_recompiles_id, retired_precompile_entries);
    }
  }
}

void _load_precompile_entry(
    const py::handle& code_obj,
    py::object guard_manager,
    py::object dynamo_code,
    int64_t isolate_recompiles_id) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  if (extra == nullptr) {
    extra = init_and_set_extra_state(code);
  }
  CacheLock lock(extra->cache_mutex);
  extra->precompile_entries.push_back(
      std::make_shared<PrecompileEntry>(
          std::move(guard_manager),
          std::move(dynamo_code),
          isolate_recompiles_id));
}

void _set_lru_cache(const py::object& boolean) {
  if (py::cast<bool>(boolean)) {
    use_lru = true;
  } else {
    use_lru = false;
  }
}

py::list _debug_get_precompile_entries(const py::handle& code_obj) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  py::list result;
  if (extra != nullptr) {
    CacheLock lock(extra->cache_mutex);
    for (const PrecompileEntryPtr& entry : extra->precompile_entries) {
      result.append(py::cast(PrecompileEntrySnapshot(*entry)));
    }
  }
  return result;
}

#include <algorithm>
#include <array>
#include <mutex>
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
bool use_lru = true;

// Strategy tokens come from a process-wide counter so that resetting one code
// object's ExtraState cannot make a stale owner's token look current again.
// Only strategy WRITES touch this, so its mutex is off the per-frame read path,
// which locks the ExtraState's own strategy_mutex instead.
uint64_t next_strategy_generation = 0;
std::mutex generation_mutex;

uint64_t next_generation() {
  std::lock_guard<std::mutex> lock(generation_mutex);
  return ++next_strategy_generation;
}

// Acquiring a mutex while holding the GIL deadlocks against a thread that holds
// the mutex and needs the GIL. Nothing under cache_mutex runs Python or drops
// the GIL (see ExtraState), so contention should not arise; if it does anyway,
// take the uncontended fast path without touching the GIL and release it before
// blocking so the owner can finish rather than deadlock.
class CacheLock {
 public:
  explicit CacheLock(std::mutex& mutex) : lock_(mutex, std::try_to_lock) {
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

  CacheLock(const CacheLock&) = delete;
  CacheLock& operator=(const CacheLock&) = delete;
  CacheLock(CacheLock&&) = delete;
  CacheLock& operator=(CacheLock&&) = delete;
  ~CacheLock() = default;

 private:
  std::unique_lock<std::mutex> lock_;
};

// What guard evaluation needs from an entry, copied under cache_mutex so the
// lock can be dropped while guards run. guard_manager owns root_mgr and
// diff_guard_root_mgr, and a concurrent invalidate() replaces it.
struct GuardCandidate {
  py::object guard_manager;
  py::object code;
  py::object backend;
  void* root_mgr;
  void* diff_guard_root_mgr;
};

// What the code object's extra slot points at. Slot reads and writes are
// serialized by the GIL; this design is not free-threading safe.
//
// Teardown protocol: destroy_extra_state marks the holder `destroying`, drops
// its reference, and deletes the holder only in that outermost call. Dropping
// the last reference runs ~ExtraState, whose Python (weakref callbacks on the
// transformed code) can reach this same code object while CPython still has
// the holder in the slot -- CPython nulls the slot only after the freefunc
// returns. During that window get_extra_state reads "no state", a nested
// reset_extra_state or freefunc call is a no-op, and init_and_set_extra_state
// hands out a detached state without touching the slot: anything installed
// there would be dropped unfreed when the outer freefunc returns.
struct ExtraStateHolder {
  ExtraStateRef state;
  bool destroying{false};
};
} // namespace

Py_ssize_t extra_index = -1;

ExtraState::ExtraState() = default;

// NOLINTNEXTLINE(bugprone-exception-escape)
ExtraState::~ExtraState() {
  // The last reference is gone, so nothing else can be inside cache_mutex. The
  // containers are detached under it and destroyed after it is released, so
  // the Python that ~CacheEntry runs never sees a locked or half-torn state.
  std::list<PrecompileEntry> precompile;
  std::list<PrecompileEntry> dead_precompile;
  std::unordered_map<int64_t, std::list<CacheEntry>> entries;
  std::list<CacheEntry> dead;
  {
    CacheLock lock(this->cache_mutex);
    precompile = std::move(this->precompile_entries);
    dead_precompile = std::move(this->precompile_graveyard);
    entries = std::move(this->cache_entry_map);
    dead = std::move(this->graveyard);
  }
}

std::list<CacheEntry>& ExtraState::cache_entry_list(
    int64_t isolate_recompiles_id) {
  return this->cache_entry_map[isolate_recompiles_id];
}

bool ExtraState::has_relevant_entries(int64_t isolate_recompiles_id) const {
  CacheLock lock(this->cache_mutex);
  return this->cache_entry_map.count(isolate_recompiles_id) > 0 ||
      (isolate_recompiles_id >= 0 && this->cache_entry_map.count(-1) > 0);
}

CacheEntry* ExtraState::find_entry(
    int64_t isolate_recompiles_id,
    PyObject* guard_manager) {
  auto it = this->cache_entry_map.find(isolate_recompiles_id);
  if (it == this->cache_entry_map.end()) {
    return nullptr;
  }
  for (CacheEntry& entry : it->second) {
    if (entry.guard_manager.ptr() == guard_manager) {
      return &entry;
    }
  }
  return nullptr;
}

CacheEntry* ExtraState::find_entry(PyObject* guard_manager) {
  for (const auto& kv : this->cache_entry_map) {
    if (CacheEntry* found = this->find_entry(kv.first, guard_manager)) {
      return found;
    }
  }
  return nullptr;
}

void ExtraState::move_to_front(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  auto& list = this->cache_entry_map[cache_entry->_isolate_recompiles_id];
  list.splice(list.begin(), list, cache_entry->_owner_loc);
}

void ExtraState::move_to_back(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);
  CHECK(cache_entry == &*cache_entry->_owner_loc);
  auto& list = this->cache_entry_map[cache_entry->_isolate_recompiles_id];
  list.splice(list.end(), list, cache_entry->_owner_loc);
}

void ExtraState::invalidate(
    CacheEntry* cache_entry,
    py::object deleted_guard_manager,
    py::object live_guard_manager) {
  CacheEntry::Detached detached;
  {
    CacheLock lock(this->cache_mutex);
    // cache_entry arrives as a non-owning raw pointer, read off the guard
    // manager before the lock was taken (CheckFunctionManager.invalidate), so
    // another thread may have cleared it while we blocked. Only a live node
    // found by identity is dereferenced.
    CacheEntry* live_entry = this->find_entry(live_guard_manager.ptr());
    if (live_entry == nullptr) {
      return;
    }
    CHECK(live_entry == cache_entry);
    detached = live_entry->invalidate(std::move(deleted_guard_manager));
    // Move the cache entry to the end of the list because these will always
    // return False.
    this->move_to_back(live_entry);
  }
  // Keep the current pointer alive but make the fields as if no-op. The
  // detached code and backend die with `detached`, after the lock.
  detached.guard_manager.attr("cache_entry") = py::none();
  detached.guard_manager.attr("extra_state") = py::none();
}

void ExtraStateHandle::invalidate(
    CacheEntry* cache_entry,
    py::object deleted_guard_manager,
    py::object live_guard_manager) {
  ExtraStateRef live = this->state.lock();
  if (live == nullptr) {
    return;
  }
  live->invalidate(
      cache_entry,
      std::move(deleted_guard_manager),
      std::move(live_guard_manager));
}

CachePin::CachePin(ExtraStateRef state) : state_(std::move(state)) {
  CacheLock lock(state_->cache_mutex);
  ++state_->pinned;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
CachePin::~CachePin() {
  std::list<CacheEntry> dead;
  std::list<PrecompileEntry> dead_precompile;
  {
    CacheLock lock(state_->cache_mutex);
    if (--state_->pinned == 0) {
      dead = std::move(state_->graveyard);
      dead_precompile = std::move(state_->precompile_graveyard);
    }
  }
}

CacheEntry* extract_cache_entry(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id) {
  if (extra_state == nullptr) {
    return nullptr;
  }
  CacheLock lock(extra_state->cache_mutex);
  // Search own bucket first, then fall back to default bucket (-1),
  // matching lookup() behavior.
  int64_t ids_to_search[] = {isolate_recompiles_id, -1};
  int num_ids = (isolate_recompiles_id >= 0) ? 2 : 1;

  for (int i = 0; i < num_ids; i++) {
    auto it = extra_state->cache_entry_map.find(ids_to_search[i]);
    if (it != extra_state->cache_entry_map.end() && !it->second.empty()) {
      return &it->second.front();
    }
  }
  return nullptr;
}

FrameState* extract_frame_state(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id) {
  if (extra_state == nullptr) {
    return nullptr;
  }
  PyObject* frame_state = nullptr;
  if (isolate_recompiles_id < 0) {
    frame_state = extra_state->frame_state.ptr();
    Py_INCREF(frame_state);
  } else {
    // Nothing that can execute Python may run under this plain mutex. Unlike
    // cache_mutex it has no CacheLock, so a thread that drops the GIL while
    // holding it wedges every other GIL-holding thread that then blocks here --
    // and operator[] default-constructs the py::dict, whose PyDict_New can
    // trigger a gen-0 collection that runs an arbitrary __del__. So the dict
    // for a new region is built BEFORE the lock and the lock only finds or
    // emplaces; `fresh` outlives the lock scope so its decref, when the key
    // already existed, also lands outside.
    py::dict fresh;
    {
      std::lock_guard<std::mutex> lock(extra_state->region_frame_state_mutex);
      auto it = extra_state->region_frame_state_map.find(isolate_recompiles_id);
      if (it == extra_state->region_frame_state_map.end()) {
        it = extra_state->region_frame_state_map
                 .emplace(isolate_recompiles_id, std::move(fresh))
                 .first;
      }
      frame_state = it->second.ptr();
      Py_INCREF(frame_state);
    }
  }
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

static ExtraStateHolder* get_extra_state_holder(PyCodeObject* code) {
  ExtraStateHolder* holder = nullptr;
  _PyCode_GetExtra((PyObject*)code, extra_index, (void**)&holder);
  return holder;
}

ExtraStateRef get_extra_state(PyCodeObject* code) {
  ExtraStateHolder* holder = get_extra_state_holder(code);
  // A destroying holder's state has already been moved out, so it reads as
  // "no ExtraState" rather than as a state mid-destruction.
  return holder == nullptr ? nullptr : holder->state;
}

void destroy_extra_state(void* obj) {
  // code_dealloc calls the freefunc for every index, including a slot that
  // reset_extra_state already cleared to NULL.
  if (obj == nullptr) {
    return;
  }
  ExtraStateHolder* holder = static_cast<ExtraStateHolder*>(obj);
  if (holder->destroying) {
    return;
  }
  holder->destroying = true;
  ExtraStateRef state = std::move(holder->state);
  state.reset();
  delete holder;
}

void reset_extra_state(PyCodeObject* code) {
  ExtraStateHolder* holder = get_extra_state_holder(code);
  if (holder == nullptr || holder->destroying) {
    return;
  }
  _PyCode_SetExtra((PyObject*)code, extra_index, nullptr);
}

ExtraStateRef init_and_set_extra_state(PyCodeObject* code) {
  ExtraStateRef state = std::make_shared<ExtraState>();
  ExtraStateHolder* holder = get_extra_state_holder(code);
  if (holder != nullptr) {
    // A live holder means the caller skipped get_extra_state.
    CHECK(holder->destroying);
    return state;
  }
  // freed by destroy_extra_state (since we need to pass these objects to C)
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  _PyCode_SetExtra(
      (PyObject*)code, extra_index, new ExtraStateHolder{state, false});
  return state;
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

static bool has_no_guards(
    void* root_mgr,
    void* diff_guard_root_mgr,
    bool is_skip_guard_eval_unsafe) {
  if (is_skip_guard_eval_unsafe && diff_guard_root_mgr != nullptr) {
    return torch::dynamo::root_guard_manager_has_no_guards(diff_guard_root_mgr);
  }
  return torch::dynamo::root_guard_manager_has_no_guards(root_mgr);
}

// Run the guards of candidates[start:], in cache order.
// Returns the index of the first match, or candidates.size() if none.
// Sets *guard_error = true if a guard evaluation exception occurred.
static size_t match_in_list(
    const std::vector<GuardCandidate>& candidates,
    size_t start,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    bool is_skip_guard_eval_unsafe,
    bool* guard_error) {
  for (size_t index = start; index < candidates.size(); ++index) {
    const GuardCandidate& candidate = candidates[index];
    bool valid =
        Py_IsFalse(backend) || backend_match(candidate.backend.ptr(), backend);

    if (valid) {
      try {
        if (is_skip_guard_eval_unsafe) {
          valid = has_no_guards(
                      candidate.root_mgr,
                      candidate.diff_guard_root_mgr,
                      /*is_skip_guard_eval_unsafe=*/true) ||
              torch::dynamo::run_root_guard_manager(
                      candidate.diff_guard_root_mgr, f_locals);
        } else {
          valid = torch::dynamo::run_root_guard_manager(
              candidate.root_mgr, f_locals);
        }
      } catch (py::error_already_set& e) {
        if (guard_error_hook) {
          py::handle guard_error_hook_handle(guard_error_hook);
          py::handle f_locals_dict = (PyObject*)f_locals->to_dict();
          guard_error_hook_handle(
              candidate.guard_manager,
              candidate.code,
              f_locals_dict,
              index,
              index == candidates.size() - 1);
        }
        e.restore();
        *guard_error = true;
        return candidates.size();
      }
    }
    if (valid) {
      return index;
    }
  }
  return candidates.size();
}

// Runs under cache_mutex, so backends are compared by identity only: a
// different object may still be __eq__-equal (two torch.compile() wrappers with
// the same options), and deciding that is user Python, which lookup() runs with
// the lock released.
static bool try_lookup_without_guard_eval_in_list(
    std::list<CacheEntry>& entries,
    PyObject* backend,
    bool is_skip_guard_eval_unsafe,
    CacheEntry** found) {
  for (CacheEntry& cache_entry : entries) {
    if (!Py_IsFalse(backend) && cache_entry.backend.ptr() != backend) {
      return false;
    }
    if (!PyCode_Check(cache_entry.code.ptr())) {
      continue;
    }
    if (has_no_guards(
            cache_entry.root_mgr,
            cache_entry.diff_guard_root_mgr,
            is_skip_guard_eval_unsafe)) {
      *found = &cache_entry;
      return true;
    }
    return false;
  }
  return true;
}

void lookup(
    ExtraState* extra_state,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    LookupResult* result,
    bool is_skip_guard_eval_unsafe) {
  // Search own bucket first, then fall back to default bucket (-1).
  // This lets isolated compiles reuse compilations from non-isolated
  // torch.compile() calls (BC friendly). New entries are still written
  // to the isolated bucket.
  std::array<int64_t, 2> ids_to_search = {isolate_recompiles_id, -1};
  size_t num_ids = (isolate_recompiles_id >= 0) ? 2 : 1;
  std::vector<GuardCandidate> precompile_candidates;
  std::array<std::vector<GuardCandidate>, 2> candidates;
  {
    CacheLock lock(extra_state->cache_mutex);
    // Precompile entries match their OWN region only, deliberately unlike the
    // cache-entry fallback. The identity guards that would tell two artifacts
    // of one model apart are exactly the ones precompile has to drop, so a
    // fallback here serves another artifact's graph for a call this region
    // does not cover, instead of the miss that serving() turns into a loud
    // error. The cache-entry fallback is narrower than it looks but is not a
    // precedent: match_in_list also requires backend_match, though note that
    // short-circuits when the backend is Py_False, which is every frame under
    // run-only. Callers that install for an isolated region must pass its id
    // (see CompilePackage.install) rather than rely on the default bucket.
    for (const PrecompileEntry& entry : extra_state->precompile_entries) {
      if (entry.isolate_recompiles_id == isolate_recompiles_id) {
        precompile_candidates.push_back(
            {entry.guard_manager,
             entry.code,
             py::object(),
             entry.root_mgr,
             nullptr});
      }
    }
    for (size_t i = 0; i < num_ids; i++) {
      auto it = extra_state->cache_entry_map.find(ids_to_search[i]);
      if (it == extra_state->cache_entry_map.end() || it->second.empty()) {
        continue;
      }
      candidates[i].reserve(it->second.size());
      for (const CacheEntry& entry : it->second) {
        candidates[i].push_back(
            {entry.guard_manager,
             entry.code,
             entry.backend,
             entry.root_mgr,
             entry.diff_guard_root_mgr});
      }
    }
  }

  // Guards run with cache_mutex released: they call user Python, which can
  // reach another compiled function and take its lock, so holding ours here
  // would let two threads deadlock on opposite orders. The match is then
  // re-found by guard manager identity; an entry unloaded or invalidated while
  // its guards ran is skipped and the search resumes with the next candidate.
  // Precompile candidates already passed their region and backend filters, so
  // match_in_list runs their guards only.
  for (size_t index = 0; index < precompile_candidates.size();) {
    bool guard_error = false;
    index = match_in_list(
        precompile_candidates,
        index,
        f_locals,
        Py_False,
        /*is_skip_guard_eval_unsafe=*/false,
        &guard_error);
    if (guard_error) {
      result->code = py::object();
      return;
    }
    if (index == precompile_candidates.size()) {
      break;
    }
    PyObject* winner = precompile_candidates[index++].guard_manager.ptr();
    CacheLock lock(extra_state->cache_mutex);
    for (const PrecompileEntry& entry : extra_state->precompile_entries) {
      if (entry.guard_manager.ptr() == winner) {
        result->code = entry.code;
        return;
      }
    }
  }

  for (size_t i = 0; i < num_ids; i++) {
    for (size_t index = 0; index < candidates[i].size();) {
      bool guard_error = false;
      index = match_in_list(
          candidates[i],
          index,
          f_locals,
          backend,
          is_skip_guard_eval_unsafe,
          &guard_error);
      if (guard_error) {
        result->code = py::object();
        return;
      }
      if (index == candidates[i].size()) {
        break;
      }
      PyObject* winner = candidates[i][index++].guard_manager.ptr();
      CacheLock lock(extra_state->cache_mutex);
      CacheEntry* live = extra_state->find_entry(ids_to_search[i], winner);
      if (live == nullptr) {
        continue;
      }
      if (use_lru) {
        extra_state->move_to_front(live);
      }
      result->code = live->code;
      result->trace_annotation = live->trace_annotation;
      return;
    }
  }
  result->code = py::none();
}

bool try_lookup_without_guard_eval(
    ExtraState* extra_state,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    LookupResult* result,
    bool is_skip_guard_eval_unsafe) {
  CacheLock lock(extra_state->cache_mutex);
  // Own region only, matching lookup().
  const PrecompileEntry* first_precompile_entry = nullptr;
  for (const auto& entry : extra_state->precompile_entries) {
    if (entry.isolate_recompiles_id == isolate_recompiles_id) {
      first_precompile_entry = &entry;
      break;
    }
  }
  std::array<int64_t, 2> ids_to_search = {isolate_recompiles_id, -1};
  size_t num_ids = (isolate_recompiles_id >= 0) ? 2 : 1;
  if (first_precompile_entry != nullptr) {
    // Only the first precompile entry can be safely fast-pathed: a later
    // guardless entry must not preempt an earlier guarded entry whose guards
    // may pass.
    if (torch::dynamo::root_guard_manager_has_no_guards(
            first_precompile_entry->root_mgr)) {
      result->code = first_precompile_entry->code;
      return true;
    }
    return false;
  }

  CacheEntry* found = nullptr;
  for (size_t i = 0; i < num_ids && found == nullptr; i++) {
    auto it = extra_state->cache_entry_map.find(ids_to_search[i]);
    if (it != extra_state->cache_entry_map.end()) {
      if (!try_lookup_without_guard_eval_in_list(
              it->second, backend, is_skip_guard_eval_unsafe, &found)) {
        return false;
      }
    }
  }

  if (found) {
    if (use_lru) {
      extra_state->move_to_front(found);
    }
    result->code = found->code;
    result->trace_annotation = found->trace_annotation;
    return true;
  }

  result->code = py::none();
  return true;
}

LookupResult create_cache_entry(
    const ExtraStateRef& extra_state,
    PyObject* guarded_code,
    PyObject* backend) {
  // Built as a one-node list before the lock -- the constructor reads
  // guarded_code's attributes in Python -- and spliced in under it, so the node
  // never moves and the wrapper below stays valid.
  std::list<CacheEntry> fresh;
  fresh.emplace_back(guarded_code, backend);
  CacheEntry& entry = fresh.front();
  int64_t id = get_current_isolate_recompiles_id();
  py::object guard_manager = entry.guard_manager;
  py::object entry_obj = py::cast(&entry, py::return_value_policy::reference);
  py::object handle = py::cast(ExtraStateHandle{extra_state});
  LookupResult result{entry.code, entry.trace_annotation};
  {
    CacheLock lock(extra_state->cache_mutex);
    entry._owner = extra_state.get();
    entry._owner_loc = fresh.begin();
    entry._isolate_recompiles_id = id;
    auto& entries = extra_state->cache_entry_list(id);
    entries.splice(use_lru ? entries.begin() : entries.end(), fresh);
    extra_state->total_cache_entry_count++;
  }
  // Set guard_manager references to extra_state and CacheEntry.
  // Warning: lifetime is controlled by C++! A clear on another thread between
  // the unlock and here leaves cache_entry pointing at a freed node; readers
  // (ExtraState::invalidate) never dereference it without first re-finding the
  // entry by guard manager identity.
  guard_manager.attr("cache_entry") = std::move(entry_obj);
  guard_manager.attr("extra_state") = std::move(handle);
  return result;
}

py::list _debug_get_cache_entry_list(const py::handle& code_obj) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraStateRef extra = get_extra_state(code);
  py::list result;
  if (extra == nullptr) {
    return result;
  }
  // The wrappers are built after the lock (Python allocation); the pin keeps
  // the nodes allocated meanwhile.
  CachePin pin(extra);
  std::vector<CacheEntry*> entries;
  {
    CacheLock lock(extra->cache_mutex);
    // Sort by isolate_recompiles_id for deterministic iteration order.
    std::vector<int64_t> ids;
    ids.reserve(extra->cache_entry_map.size());
    for (auto& kv : extra->cache_entry_map) {
      ids.push_back(kv.first);
    }
    std::sort(ids.begin(), ids.end());
    entries.reserve(extra->total_cache_entry_count);
    for (int64_t id : ids) {
      for (CacheEntry& e : extra->cache_entry_map[id]) {
        entries.push_back(&e);
      }
    }
  }
  for (CacheEntry* e : entries) {
    result.append(py::cast(e, py::return_value_policy::reference));
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
  ExtraStateRef extra = get_extra_state(code);
  py::list result;
  if (extra == nullptr) {
    return result;
  }
  CachePin pin(extra);
  std::vector<CacheEntry*> entries;
  {
    CacheLock lock(extra->cache_mutex);
    auto it = extra->cache_entry_map.find(isolate_recompiles_id);
    if (it != extra->cache_entry_map.end()) {
      entries.reserve(it->second.size());
      for (CacheEntry& e : it->second) {
        entries.push_back(&e);
      }
    }
  }
  for (CacheEntry* e : entries) {
    result.append(py::cast(e, py::return_value_policy::reference));
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
  ExtraStateRef extra = get_extra_state(code);
  if (extra == nullptr) {
    return;
  }
  {
    // ~CacheEntry runs Python, so `dead` dies after the lock is released.
    std::list<CacheEntry> dead;
    {
      CacheLock lock(extra->cache_mutex);
      auto it = extra->cache_entry_map.find(isolate_recompiles_id);
      if (it != extra->cache_entry_map.end()) {
        TORCH_CHECK(extra->total_cache_entry_count >= it->second.size());
        extra->total_cache_entry_count -= it->second.size();
        if (extra->pinned > 0) {
          extra->graveyard.splice(extra->graveyard.end(), it->second);
        } else {
          dead = std::move(it->second);
        }
        extra->cache_entry_map.erase(it);
      }
    }
  }
  {
    std::lock_guard<std::mutex> lock(extra->strategy_mutex);
    extra->region_strategy_map.erase(isolate_recompiles_id);
  }
  {
    // Same rule as extract_frame_state: the dict's decref can free arbitrary
    // Python objects, so it must not happen under the plain mutex. Move it out
    // and let it die after the lock is released.
    py::dict evicted;
    {
      std::lock_guard<std::mutex> lock(extra->region_frame_state_mutex);
      auto it = extra->region_frame_state_map.find(isolate_recompiles_id);
      if (it != extra->region_frame_state_map.end()) {
        evicted = std::move(it->second);
        extra->region_frame_state_map.erase(it);
      }
    }
  }
}

size_t _get_total_cache_entry_count(const py::handle& code_obj) {
  TORCH_CHECK(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraStateRef extra = get_extra_state(code);
  if (extra == nullptr) {
    return 0;
  }
  CacheLock lock(extra->cache_mutex);
  return extra->total_cache_entry_count;
}

PrecompileEntry::PrecompileEntry(
    py::object gm,
    py::object c,
    int64_t region_id,
    py::object owner_token)
    : guard_manager(std::move(gm)),
      code(std::move(c)),
      isolate_recompiles_id(region_id),
      owner(std::move(owner_token)) {
  TORCH_CHECK(
      PyCode_Check(code.ptr()), "Expecting CodeType from PrecompileEntry.");
  root_mgr =
      torch::dynamo::convert_to_root_guard_manager(guard_manager.attr("root"));
}

// Unlinks matching entries under the lock and destroys them after it: their
// decrefs can free guard managers and run finalizers. While pinned they are
// parked in the graveyard instead, since _debug_get_precompile_entries hands
// raw wrappers to the compile callback.
template <typename Pred>
static void remove_precompile_entries_if(ExtraState* extra, Pred pred) {
  std::list<PrecompileEntry> removed;
  {
    CacheLock lock(extra->cache_mutex);
    auto& entries = extra->precompile_entries;
    auto& sink = extra->pinned > 0 ? extra->precompile_graveyard : removed;
    for (auto it = entries.begin(); it != entries.end();) {
      auto next = std::next(it);
      if (pred(*it)) {
        sink.splice(sink.end(), entries, it);
      }
      it = next;
    }
  }
}

void _reset_precompile_entries(const py::handle& code_obj) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraStateRef extra = get_extra_state(code);
  if (extra != nullptr) {
    remove_precompile_entries_if(
        extra.get(), [](const PrecompileEntry&) { return true; });
  }
}

void _reset_precompile_entries_for_region(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraStateRef extra = get_extra_state(code);
  if (extra != nullptr) {
    remove_precompile_entries_if(
        extra.get(), [isolate_recompiles_id](const PrecompileEntry& entry) {
          return entry.isolate_recompiles_id == isolate_recompiles_id;
        });
  }
}

void _reset_precompile_entries_for_owner(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id,
    const py::handle& owner) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraStateRef extra = get_extra_state(code);
  if (extra != nullptr) {
    PyObject* owner_ptr = owner.ptr();
    remove_precompile_entries_if(
        extra.get(),
        [isolate_recompiles_id, owner_ptr](const PrecompileEntry& entry) {
          return entry.isolate_recompiles_id == isolate_recompiles_id &&
              entry.owner.ptr() == owner_ptr;
        });
  }
}

void _load_precompile_entry(
    const py::handle& code_obj,
    py::object guard_manager,
    py::object dynamo_code,
    int64_t isolate_recompiles_id,
    py::object owner) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraStateRef extra = get_extra_state(code);
  if (extra == nullptr) {
    extra = init_and_set_extra_state(code);
  }
  // Built before the lock: the constructor reads guard_manager.root in Python.
  PrecompileEntry entry(
      std::move(guard_manager),
      std::move(dynamo_code),
      isolate_recompiles_id,
      std::move(owner));
  CacheLock lock(extra->cache_mutex);
  extra->precompile_entries.push_back(std::move(entry));
}

void _set_lru_cache(const py::object& boolean) {
  if (py::cast<bool>(boolean)) {
    use_lru = true;
  } else {
    use_lru = false;
  }
}

py::list _debug_get_precompile_entries(const py::handle& code_obj) {
  TORCH_CHECK_TYPE(PyCode_Check(code_obj.ptr()), "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraStateRef extra = get_extra_state(code);
  py::list result;
  if (extra == nullptr) {
    return result;
  }
  CachePin pin(extra);
  std::vector<PrecompileEntry*> entries;
  {
    CacheLock lock(extra->cache_mutex);
    entries.reserve(extra->precompile_entries.size());
    for (PrecompileEntry& e : extra->precompile_entries) {
      entries.push_back(&e);
    }
  }
  for (PrecompileEntry* e : entries) {
    result.append(py::cast(e, py::return_value_policy::reference));
  }
  return result;
}

bool _has_precompile_entries(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id) {
  TORCH_CHECK_TYPE(PyCode_Check(code_obj.ptr()), "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraStateRef extra = get_extra_state(code);
  if (extra == nullptr) {
    return false;
  }
  // Region exact, matching lookup(): an entry from another region never serves
  // this one, so a second artifact installed on the same code object is not
  // coverage for the first. A loaded artifact runs this on every served call,
  // hence no py::list: the scan below neither allocates nor runs Python.
  CacheLock lock(extra->cache_mutex);
  for (const PrecompileEntry& entry : extra->precompile_entries) {
    if (entry.isolate_recompiles_id == isolate_recompiles_id) {
      return true;
    }
  }
  return false;
}

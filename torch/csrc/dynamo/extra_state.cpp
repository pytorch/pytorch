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
// the mutex and needs the GIL, and lookup() does exactly that: guard evaluation
// runs Python (LAMBDA_GUARD calls back into the interpreter), which can drop
// the GIL at any bytecode boundary. Take the uncontended fast path without
// touching the GIL, and release it before blocking so the owner can finish.
class CacheLock {
 public:
  explicit CacheLock(std::recursive_mutex& mutex)
      : lock_(mutex, std::try_to_lock) {
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
  std::unique_lock<std::recursive_mutex> lock_;
};

// Marks the window in which a cache_mutex holder runs Python (guard
// evaluation, backend __eq__, pybind attribute stores). cache_mutex is
// recursive, so that Python can re-enter this state on the same thread --
// reset_code from a guard, most notably -- and clear_in_place consults the
// depth to know it must defer node destruction. Caller must hold cache_mutex.
class CachePythonDepth {
 public:
  explicit CachePythonDepth(ExtraState* state) : state_(state) {
    ++state_->cache_python_depth;
  }
  ~CachePythonDepth() {
    --state_->cache_python_depth;
  }

  CachePythonDepth(const CachePythonDepth&) = delete;
  CachePythonDepth& operator=(const CachePythonDepth&) = delete;
  CachePythonDepth(CachePythonDepth&&) = delete;
  CachePythonDepth& operator=(CachePythonDepth&&) = delete;

 private:
  ExtraState* state_;
};
} // namespace

Py_ssize_t extra_index = -1;

ExtraState::ExtraState(PyCodeObject* orig_code_arg)
    : orig_code(orig_code_arg) {}

std::list<CacheEntry>& ExtraState::cache_entry_list(
    int64_t isolate_recompiles_id) {
  return this->cache_entry_map[isolate_recompiles_id];
}

bool ExtraState::has_any_cache_entries() const {
  CacheLock lock(this->cache_mutex);
  return this->total_cache_entry_count > 0;
}

bool ExtraState::has_relevant_entries(int64_t isolate_recompiles_id) const {
  CacheLock lock(this->cache_mutex);
  return this->cache_entry_map.count(isolate_recompiles_id) > 0 ||
      (isolate_recompiles_id >= 0 && this->cache_entry_map.count(-1) > 0);
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

void ExtraState::invalidate_locked(
    const py::object& deleted_guard_manager,
    const py::object& live_guard_manager) {
  // Locate the entry by the IDENTITY of the guard manager that owns it, never
  // by an address handed in: a std::list node is recycled, so a fresh entry
  // allocated at the same address would pass an address check and then be
  // wrongly invalidated, killing a valid compilation. Only live nodes are
  // dereferenced here, so an already-destroyed entry is never touched.
  CacheEntry* live_entry = nullptr;
  for (auto& [id, entries] : this->cache_entry_map) {
    (void)id;
    for (CacheEntry& live : entries) {
      if (live.guard_manager.ptr() == live_guard_manager.ptr()) {
        live_entry = &live;
        break;
      }
    }
    if (live_entry != nullptr) {
      break;
    }
  }
  if (live_entry != nullptr) {
    CHECK(live_entry->_owner == this);
    CHECK(live_entry == &*live_entry->_owner_loc);
    live_entry->invalidate(py::object(deleted_guard_manager));
    // Move the cache entry to the end of the list because these will always
    // return False.
    this->move_to_back(live_entry);
  }
}

void ExtraState::drain_pending_invalidations() {
  if (!this->has_pending_invalidations.load(std::memory_order_acquire)) {
    return;
  }
  // Swap out under the pending mutex: applying an invalidation can run
  // arbitrary Python (dropping a guard manager reference), which must not
  // happen while that mutex is held.
  std::vector<std::pair<py::object, py::object>> pending;
  {
    std::lock_guard<std::mutex> lock(this->pending_invalidation_mutex);
    pending.swap(this->pending_invalidations);
    this->has_pending_invalidations.store(false, std::memory_order_release);
  }
  for (auto& [deleted_gm, live_gm] : pending) {
    this->invalidate_locked(deleted_gm, live_gm);
  }
}

void ExtraState::park_eviction(PendingEviction eviction) {
  std::lock_guard<std::mutex> lock(this->pending_invalidation_mutex);
  this->pending_evictions.push_back(std::move(eviction));
  this->has_pending_evictions.store(true, std::memory_order_release);
}

void ExtraState::apply_pending_evictions(
    std::list<PrecompileEntry>& dead_precompile,
    std::unordered_map<int64_t, std::list<CacheEntry>>& dead_cache) {
  if (this->cache_python_depth != 0) {
    // A lookup below this frame on this thread holds iterators into these
    // lists; the next depth-zero holder applies the evictions.
    return;
  }
  if (!this->has_pending_evictions.load(std::memory_order_acquire)) {
    return;
  }
  std::vector<PendingEviction> evictions;
  {
    std::lock_guard<std::mutex> lock(this->pending_invalidation_mutex);
    evictions.swap(this->pending_evictions);
    this->has_pending_evictions.store(false, std::memory_order_release);
  }
  for (auto& eviction : evictions) {
    switch (eviction.kind) {
      case PendingEviction::CLEAR_ALL: {
        dead_precompile.splice(dead_precompile.end(), this->precompile_entries);
        for (auto& [id, entries] : this->cache_entry_map) {
          auto& dst = dead_cache[id];
          dst.splice(dst.end(), entries);
        }
        this->cache_entry_map.clear();
        this->total_cache_entry_count = 0;
        break;
      }
      case PendingEviction::PRECOMPILE_ALL: {
        dead_precompile.splice(dead_precompile.end(), this->precompile_entries);
        break;
      }
      case PendingEviction::CACHE_REGION: {
        auto it = this->cache_entry_map.find(eviction.region_id);
        if (it != this->cache_entry_map.end()) {
          this->total_cache_entry_count -= it->second.size();
          auto& dst = dead_cache[eviction.region_id];
          dst.splice(dst.end(), it->second);
          this->cache_entry_map.erase(it);
        }
        break;
      }
      case PendingEviction::OWNER:
      case PendingEviction::PRECOMPILE_REGION: {
        bool by_owner = eviction.kind == PendingEviction::OWNER;
        auto& entries = this->precompile_entries;
        for (auto it = entries.begin(); it != entries.end();) {
          auto next = std::next(it);
          if (it->isolate_recompiles_id == eviction.region_id &&
              (!by_owner || it->owner.ptr() == eviction.owner.ptr())) {
            dead_precompile.splice(dead_precompile.end(), entries, it);
          }
          it = next;
        }
        break;
      }
    }
  }
}

void ExtraState::invalidate(
    CacheEntry* cache_entry,
    py::object deleted_guard_manager,
    py::object live_guard_manager) {
  // cache_entry arrives as a non-owning raw pointer, read off the guard
  // manager before any lock was taken (CheckFunctionManager.invalidate);
  // another thread can destroy it at any point, so it is never dereferenced
  // -- the live entry is re-located by guard manager identity instead.
  (void)cache_entry;
  // Sometimes setting the cache_entry->code to None causes the orig_code to be
  // freed. This calls destroy_extra_state, which deletes the extra_state and
  // all the cache_entries. This causes the `this` pointer to be a dangling
  // pointer, causing a segfault. So, we manually inc/dec ref the original code
  // pointer to prevent triggering of destroy_extra_state while the invalidate
  // function is running.
  Py_INCREF(this->orig_code);
  {
    std::unique_lock<std::recursive_mutex> lock(
        this->cache_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
      // NEVER block on cache_mutex here: invalidate is reached from
      // weakref.finalize, which GC can fire during guard evaluation while
      // ANOTHER ExtraState's cache_mutex is held. Two threads doing that
      // against each other's states would deadlock -- CacheLock releases only
      // the GIL, not the peer's lock. Park the request instead; the next
      // holder of cache_mutex (lookup, or a later invalidate) applies it.
      std::lock_guard<std::mutex> pending_lock(
          this->pending_invalidation_mutex);
      this->pending_invalidations.emplace_back(
          std::move(deleted_guard_manager), std::move(live_guard_manager));
      this->has_pending_invalidations.store(true, std::memory_order_release);
    } else {
      this->drain_pending_invalidations();
      this->invalidate_locked(deleted_guard_manager, live_guard_manager);
    }
  }
  // The lock must be released BEFORE the decref: if this drops the last
  // reference, destroy_extra_state deletes `this` along with cache_mutex, and
  // unlocking a destroyed mutex is undefined behaviour.
  Py_DECREF(this->orig_code);
}

void ExtraState::clear_in_place() {
  // Destructors of the evicted containers run AFTER the locks release: a
  // CacheEntry / PrecompileEntry destructor drops py::objects, and any Python
  // that runs from those decrefs must not re-enter this state's containers
  // mid-mutation or block behind a mutex this thread still holds.
  std::list<PrecompileEntry> dead_precompile_entries;
  std::unordered_map<int64_t, std::list<CacheEntry>> dead_cache_entries;
  std::vector<std::pair<py::object, py::object>> dead_pending;
  std::vector<PendingEviction> dead_evictions;
  py::dict dead_frame_state;
  std::unordered_map<int64_t, py::dict> dead_region_frame_state;
  {
    CacheLock lock(this->cache_mutex);
    if (this->cache_python_depth > 0) {
      // This thread is INSIDE lookup()'s guard evaluation (the recursive
      // cache_mutex is how we got here, via Python run by a guard reaching
      // reset_code): that lookup holds live iterators into these lists, so
      // neither destroying nor even relinking their nodes is safe. Park the
      // clear; the next depth-zero cache_mutex holder applies it. The clear
      // landing "just after" the interrupted lookup matches the pre-existing
      // asynchrony of a reset racing a lookup from another thread.
      this->park_eviction(
          PendingEviction{PendingEviction::CLEAR_ALL, -1, py::none()});
    } else {
      dead_precompile_entries.swap(this->precompile_entries);
      dead_cache_entries.swap(this->cache_entry_map);
      this->total_cache_entry_count = 0;
      // Nothing a parked eviction could still remove survives this clear.
      // Swapped out, destroyed after the locks release like everything else
      // here: an owner's decref may run Python.
      std::lock_guard<std::mutex> pending(this->pending_invalidation_mutex);
      dead_evictions.swap(this->pending_evictions);
      this->has_pending_evictions.store(false, std::memory_order_release);
    }
  }
  {
    std::lock_guard<std::mutex> lock(this->pending_invalidation_mutex);
    dead_pending.swap(this->pending_invalidations);
  }
  // frame_state itself is only touched under the GIL (see
  // extract_frame_state), which the caller holds. The fresh dict is built
  // BEFORE the member moves: PyDict_New can trigger a gen-0 collection whose
  // finalizers run arbitrary Python (or drop the GIL), and in that window
  // extract_frame_state would Py_INCREF a moved-from null pointer.
  py::dict fresh_frame_state;
  dead_frame_state = std::move(this->frame_state);
  this->frame_state = std::move(fresh_frame_state);
  {
    std::lock_guard<std::mutex> lock(this->region_frame_state_mutex);
    dead_region_frame_state.swap(this->region_frame_state_map);
  }
  {
    std::lock_guard<std::mutex> lock(this->strategy_mutex);
    this->strategy = FrameExecStrategy{DEFAULT, DEFAULT};
    // A fresh token, NOT zero: zero is what get_exec_strategy_token returns
    // for a never-written state, so resetting to it would let a holder of
    // that pre-write token win compare_and_set after an intervening write
    // plus reset -- exactly the staleness the generation exists to refuse.
    this->strategy_generation = next_generation();
    this->region_strategy_map.clear();
  }
}

CacheEntry* extract_cache_entry(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id) {
  if (extra_state == nullptr) {
    return nullptr;
  }
  // Reaped nodes die AFTER the lock releases (locals declared before it).
  std::list<PrecompileEntry> reaped_precompile;
  std::unordered_map<int64_t, std::list<CacheEntry>> reaped_cache;
  CacheLock lock(extra_state->cache_mutex);
  extra_state->apply_pending_evictions(reaped_precompile, reaped_cache);
  extra_state->drain_pending_invalidations();
  // Search own bucket first, then fall back to default bucket (-1),
  // matching lookup() behavior.
  std::array<int64_t, 2> ids_to_search = {isolate_recompiles_id, -1};
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

ExtraState* get_extra_state(PyCodeObject* code) {
  ExtraState* extra = nullptr;
  _PyCode_GetExtra((PyObject*)code, extra_index, (void**)&extra);
  return extra;
}

void destroy_extra_state(void* obj) {
  ExtraState* extra = (ExtraState*)obj;
  delete extra;
}

void reset_extra_state(PyCodeObject* code) {
  ExtraState* extra = get_extra_state(code);
  if (extra != nullptr) {
    extra->clear_in_place();
  }
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
    const CacheEntry& cache_entry,
    bool is_skip_guard_eval_unsafe) {
  if (is_skip_guard_eval_unsafe && cache_entry.diff_guard_root_mgr != nullptr) {
    return torch::dynamo::root_guard_manager_has_no_guards(
        cache_entry.diff_guard_root_mgr);
  }
  return torch::dynamo::root_guard_manager_has_no_guards(cache_entry.root_mgr);
}

// Search a region's cache list for a matching entry.
// Returns the matching CacheEntry, or nullptr if no match.
// Sets *guard_error = true if a guard evaluation exception occurred.
static CacheEntry* lookup_in_list(
    std::list<CacheEntry>& entries,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    bool is_skip_guard_eval_unsafe,
    bool* guard_error,
    PyObject** maybe_cached_code) {
  size_t index = 0;
  for (CacheEntry& cache_entry : entries) {
    bool valid = Py_IsFalse(backend) ||
        backend_match(cache_entry.backend.ptr(), backend);

    if (valid) {
      try {
        if (is_skip_guard_eval_unsafe) {
          valid = cache_entry_has_no_guards(
                      cache_entry, /*is_skip_guard_eval_unsafe=*/true) ||
              torch::dynamo::run_root_guard_manager(
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
              index == entries.size() - 1);
        }
        e.restore();
        *maybe_cached_code = nullptr;
        *guard_error = true;
        return nullptr;
      }
    }
    if (valid) {
      return &cache_entry;
    }
    ++index;
  }
  return nullptr;
}

static bool try_lookup_without_guard_eval_in_list(
    std::list<CacheEntry>& entries,
    PyObject* backend,
    bool is_skip_guard_eval_unsafe,
    CacheEntry** found) {
  for (CacheEntry& cache_entry : entries) {
    bool valid = Py_IsFalse(backend) ||
        backend_match(cache_entry.backend.ptr(), backend);

    if (valid) {
      if (!PyCode_Check(cache_entry.code.ptr())) {
        continue;
      }
      if (cache_entry_has_no_guards(cache_entry, is_skip_guard_eval_unsafe)) {
        *found = &cache_entry;
        return true;
      }
      return false;
    }
  }
  return true;
}

void lookup(
    ExtraState* extra_state,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    PyObject** maybe_cached_code,
    std::string* trace_annotation,
    bool is_skip_guard_eval_unsafe) {
  // Reaped nodes die AFTER the lock releases (locals declared before it).
  std::list<PrecompileEntry> reaped_precompile;
  std::unordered_map<int64_t, std::list<CacheEntry>> reaped_cache;
  CacheLock lock(extra_state->cache_mutex);
  extra_state->apply_pending_evictions(reaped_precompile, reaped_cache);
  extra_state->drain_pending_invalidations();
  // Guard evaluation below runs Python that can re-enter this state on this
  // thread (reset_code -> clear_in_place); the depth tells clear_in_place to
  // park dead nodes instead of destroying what this frame is iterating.
  CachePythonDepth python_depth(extra_state);
  CacheEntry* found = nullptr;
  bool guard_error = false;

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
  for (const auto& entry : extra_state->precompile_entries) {
    if (entry.isolate_recompiles_id == isolate_recompiles_id &&
        torch::dynamo::run_root_guard_manager(entry.root_mgr, f_locals)) {
      *maybe_cached_code = entry.code.ptr();
      return;
    }
  }

  // Search own bucket first, then fall back to default bucket (-1).
  // This lets isolated compiles reuse compilations from non-isolated
  // torch.compile() calls (BC friendly). New entries are still written
  // to the isolated bucket.
  std::array<int64_t, 2> ids_to_search = {isolate_recompiles_id, -1};
  int num_ids = (isolate_recompiles_id >= 0) ? 2 : 1;
  std::list<CacheEntry>* found_list = nullptr;

  for (int i = 0; i < num_ids && found == nullptr; i++) {
    auto it = extra_state->cache_entry_map.find(ids_to_search[i]);
    if (it != extra_state->cache_entry_map.end()) {
      found = lookup_in_list(
          it->second,
          f_locals,
          backend,
          is_skip_guard_eval_unsafe,
          &guard_error,
          maybe_cached_code);
      if (guard_error) {
        return;
      }
      if (found) {
        found_list = &it->second;
      }
    }
  }

  if (found) {
    if (use_lru) {
      extra_state->move_to_front(found, *found_list);
    }
    *maybe_cached_code = found->code.ptr();
    *trace_annotation = found->trace_annotation;
    return;
  }
  *maybe_cached_code = py::none().ptr();
}

bool try_lookup_without_guard_eval(
    ExtraState* extra_state,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    PyObject** maybe_cached_code,
    std::string* trace_annotation,
    bool is_skip_guard_eval_unsafe) {
  // Reaped nodes die AFTER the lock releases (locals declared before it).
  std::list<PrecompileEntry> reaped_precompile;
  std::unordered_map<int64_t, std::list<CacheEntry>> reaped_cache;
  CacheLock lock(extra_state->cache_mutex);
  extra_state->apply_pending_evictions(reaped_precompile, reaped_cache);
  // A parked invalidation must not keep serving through this no-guard-eval
  // fast path (a guardless entry would never be re-checked otherwise).
  extra_state->drain_pending_invalidations();
  // backend_match below can run a backend __eq__; same re-entrancy rule as
  // lookup().
  CachePythonDepth python_depth(extra_state);
  // Own region only, matching lookup().
  const PrecompileEntry* first_precompile_entry = nullptr;
  for (const auto& entry : extra_state->precompile_entries) {
    if (entry.isolate_recompiles_id == isolate_recompiles_id) {
      first_precompile_entry = &entry;
      break;
    }
  }
  std::array<int64_t, 2> ids_to_search = {isolate_recompiles_id, -1};
  int num_ids = (isolate_recompiles_id >= 0) ? 2 : 1;
  if (first_precompile_entry != nullptr) {
    // Only the first precompile entry can be safely fast-pathed: a later
    // guardless entry must not preempt an earlier guarded entry whose guards
    // may pass.
    if (torch::dynamo::root_guard_manager_has_no_guards(
            first_precompile_entry->root_mgr)) {
      *maybe_cached_code = first_precompile_entry->code.ptr();
      return true;
    }
    return false;
  }

  std::list<CacheEntry>* found_list = nullptr;
  CacheEntry* found = nullptr;

  for (int i = 0; i < num_ids && found == nullptr; i++) {
    auto it = extra_state->cache_entry_map.find(ids_to_search[i]);
    if (it != extra_state->cache_entry_map.end()) {
      if (!try_lookup_without_guard_eval_in_list(
              it->second, backend, is_skip_guard_eval_unsafe, &found)) {
        return false;
      }
      if (found) {
        found_list = &it->second;
      }
    }
  }

  if (found) {
    if (use_lru) {
      extra_state->move_to_front(found, *found_list);
    }
    *maybe_cached_code = found->code.ptr();
    *trace_annotation = found->trace_annotation;
    return true;
  }

  *maybe_cached_code = Py_None;
  return true;
}

CacheEntry* create_cache_entry(
    ExtraState* extra_state,
    PyObject* guarded_code,
    PyObject* backend) {
  // Reaped nodes die AFTER the lock releases (locals declared before it).
  std::list<PrecompileEntry> reaped_precompile;
  std::unordered_map<int64_t, std::list<CacheEntry>> reaped_cache;
  CacheLock lock(extra_state->cache_mutex);
  extra_state->apply_pending_evictions(reaped_precompile, reaped_cache);
  extra_state->drain_pending_invalidations();
  // The pybind attribute stores below run Python; same re-entrancy rule as
  // lookup().
  CachePythonDepth python_depth(extra_state);
  int64_t id = get_current_isolate_recompiles_id();
  auto& entries = extra_state->cache_entry_list(id);
  std::list<CacheEntry>::iterator new_iter;
  if (use_lru) {
    entries.emplace_front(guarded_code, backend);
    new_iter = entries.begin();
  } else {
    entries.emplace_back(guarded_code, backend);
    new_iter = std::prev(entries.end());
  }
  new_iter->_owner = extra_state;
  new_iter->_owner_loc = new_iter;
  new_iter->_isolate_recompiles_id = id;
  extra_state->total_cache_entry_count++;
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
      for (CacheEntry& e : extra->cache_entry_map[id]) {
        result.append(py::cast(e, py::return_value_policy::reference));
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
      for (CacheEntry& e : it->second) {
        result.append(py::cast(e, py::return_value_policy::reference));
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
  {
    // Evicted entries die AFTER the lock releases: a CacheEntry destructor
    // drops py::objects whose __del__ can re-enter this state -- e.g. unload
    // another artifact on the same code object -- and under cache_mutex that
    // re-entry would mutate the container mid-erase.
    std::list<CacheEntry> evicted;
    {
      CacheLock lock(extra->cache_mutex);
      if (extra->cache_python_depth > 0) {
        // Python run BY this state's in-flight lookup got here (a guard, a
        // backend __eq__); that lookup holds iterators into this list. Park
        // the splice for the next depth-zero holder, like reset_code does.
        extra->park_eviction(ExtraState::PendingEviction{
            ExtraState::PendingEviction::CACHE_REGION,
            isolate_recompiles_id,
            py::none()});
      } else {
        auto it = extra->cache_entry_map.find(isolate_recompiles_id);
        if (it != extra->cache_entry_map.end()) {
          TORCH_CHECK(extra->total_cache_entry_count >= it->second.size());
          extra->total_cache_entry_count -= it->second.size();
          evicted = std::move(it->second);
          extra->cache_entry_map.erase(it);
        }
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
  ExtraState* extra = get_extra_state(code);
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

void _reset_precompile_entries(const py::handle& code_obj) {
  TORCH_CHECK_TYPE(
      py::isinstance(code_obj, py::module::import("types").attr("CodeType")),
      "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  if (extra != nullptr) {
    // Destroyed after the lock releases; see _clear_cache_entries_for_region.
    std::list<PrecompileEntry> evicted;
    {
      CacheLock lock(extra->cache_mutex);
      if (extra->cache_python_depth > 0) {
        // See _clear_cache_entries_for_region.
        extra->park_eviction(ExtraState::PendingEviction{
            ExtraState::PendingEviction::PRECOMPILE_ALL, -1, py::none()});
      } else {
        evicted.swap(extra->precompile_entries);
      }
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
    // Matching nodes are spliced out under the lock and destroyed after it
    // releases; see _clear_cache_entries_for_region.
    std::list<PrecompileEntry> evicted;
    {
      CacheLock lock(extra->cache_mutex);
      if (extra->cache_python_depth > 0) {
        // See _clear_cache_entries_for_region.
        extra->park_eviction(ExtraState::PendingEviction{
            ExtraState::PendingEviction::PRECOMPILE_REGION,
            isolate_recompiles_id,
            py::none()});
      } else {
        auto& entries = extra->precompile_entries;
        for (auto it = entries.begin(); it != entries.end();) {
          auto next = std::next(it);
          if (it->isolate_recompiles_id == isolate_recompiles_id) {
            evicted.splice(evicted.end(), entries, it);
          }
          it = next;
        }
      }
    }
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
  ExtraState* extra = get_extra_state(code);
  if (extra != nullptr) {
    // Matching nodes are spliced out under the lock and destroyed after it
    // releases; see _clear_cache_entries_for_region.
    std::list<PrecompileEntry> evicted;
    {
      std::unique_lock<std::recursive_mutex> lock(
          extra->cache_mutex, std::try_to_lock);
      if (!lock.owns_lock() || extra->cache_python_depth > 0) {
        // Never BLOCK here: a dead CompilePackage reaches this from
        // weakref.finalize, which GC can fire while another state's
        // cache_mutex is held -- the same ABBA cycle invalidate() parks to
        // avoid. And even uncontended, the recursive mutex admits this call
        // from Python run BY A GUARD of this very state's in-flight lookup,
        // whose iterators must see its lists untouched. Park it either way.
        extra->park_eviction(ExtraState::PendingEviction{
            ExtraState::PendingEviction::OWNER,
            isolate_recompiles_id,
            py::reinterpret_borrow<py::object>(owner)});
        return;
      }
      PyObject* owner_ptr = owner.ptr();
      auto& entries = extra->precompile_entries;
      for (auto it = entries.begin(); it != entries.end();) {
        auto next = std::next(it);
        if (it->isolate_recompiles_id == isolate_recompiles_id &&
            it->owner.ptr() == owner_ptr) {
          evicted.splice(evicted.end(), entries, it);
        }
        it = next;
      }
    }
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
  ExtraState* extra = get_extra_state(code);
  if (extra == nullptr) {
    extra = init_and_set_extra_state(code);
  }
  CacheLock lock(extra->cache_mutex);
  auto entry = PrecompileEntry(
      std::move(guard_manager),
      std::move(dynamo_code),
      isolate_recompiles_id,
      std::move(owner));
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
  ExtraState* extra = get_extra_state(code);
  py::list result;
  if (extra != nullptr) {
    // Reaped nodes die AFTER the lock releases (locals declared before it).
    std::list<PrecompileEntry> reaped_precompile;
    std::unordered_map<int64_t, std::list<CacheEntry>> reaped_cache;
    CacheLock lock(extra->cache_mutex);
    extra->apply_pending_evictions(reaped_precompile, reaped_cache);
    for (PrecompileEntry& e : extra->precompile_entries) {
      result.append(py::cast(e, py::return_value_policy::reference));
    }
  }
  return result;
}

bool _has_precompile_entries(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id) {
  TORCH_CHECK_TYPE(PyCode_Check(code_obj.ptr()), "expected a code object!");
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  ExtraState* extra = get_extra_state(code);
  if (extra == nullptr) {
    return false;
  }
  // Region exact, matching lookup(): an entry from another region never serves
  // this one, so a second artifact installed on the same code object is not
  // coverage for the first. A loaded artifact runs this on every served call,
  // hence no py::list and no Python executed under the lock -- the wait inside
  // CacheLock is the only place the GIL can drop.
  // Reaped nodes die AFTER the lock releases (locals declared before it).
  std::list<PrecompileEntry> reaped_precompile;
  std::unordered_map<int64_t, std::list<CacheEntry>> reaped_cache;
  CacheLock lock(extra->cache_mutex);
  extra->apply_pending_evictions(reaped_precompile, reaped_cache);
  for (const PrecompileEntry& entry : extra->precompile_entries) {
    if (entry.isolate_recompiles_id == isolate_recompiles_id) {
      return true;
    }
  }
  return false;
}

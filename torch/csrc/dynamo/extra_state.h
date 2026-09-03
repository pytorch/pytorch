#pragma once

#include <Python.h>

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#include <torch/csrc/dynamo/framelocals_mapping.h>

#ifdef __cplusplus

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace py = pybind11;

extern "C" {

#else

#include <stdbool.h>

#endif

enum FrameAction {
  DEFAULT, // look through the cache, compile if not found
  SKIP, // eager
  RUN_ONLY, // look through the cache, run eager if not found
};

typedef struct FrameExecStrategy {
  enum FrameAction cur_action; // action to take for current frame
  enum FrameAction recursive_action; // action to take for recursive frames
} FrameExecStrategy;

// Points to the extra scratch space on the code object
extern Py_ssize_t extra_index;

// function to call when cache lookup errors
extern PyObject* guard_error_hook;

typedef PyObject FrameState;
typedef struct CacheEntry CacheEntry;

// ExtraState encapsulates CacheEntry and FrameState. ExtraState is the highest
// level of abstraction of what is stored on the extra code object. Previously,
// we saved different parts on different extra indexes.  We prefer this way
// because of cleaner abstraction and faster SetExtra access.

#ifdef __cplusplus

typedef struct VISIBILITY_HIDDEN PrecompileEntry {
  py::object guard_manager;
  py::object code;
  void* root_mgr;
  int64_t isolate_recompiles_id;
  // Opaque token identifying who installed this entry. Several packages may
  // legitimately hold entries for one code object in one region -- a library
  // frame two loaded models both reach -- so teardown has to remove what THIS
  // installer put here and nothing else. Compared by identity; py::none() for
  // callers that do not track ownership.
  py::object owner;

  PrecompileEntry(
      py::object gm,
      py::object c,
      int64_t region_id,
      py::object owner_token);
} PrecompileEntry;

// The code object's extra slot holds one reference; every lookup, compile
// callback and pin holds another for as long as it touches the state, so
// reset_code() on one thread can only detach the state, never free it under a
// thread still inside it. The state and its mutex die with the last reference.
typedef struct VISIBILITY_HIDDEN ExtraState {
  // Declared first so it is destroyed last: everything below is only ever
  // touched under it. Nothing that can run Python -- no decref, no attribute
  // access, no Python allocation -- may happen while it is held: a finalizer
  // reaching invalidate() on this same state would self-deadlock, and a guard
  // taking another function's lock would deadlock against the opposite order.
  // Callers detach under the lock and release after it.
  mutable std::mutex cache_mutex;
  std::list<PrecompileEntry> precompile_entries;
  // Per-compile cache map: isolate_recompiles_id -> list of CacheEntry.
  // id -1 is the default (non-isolated) bucket. id >= 0 are isolated compiles.
  // All cache entries live in this map — there is no separate default list.
  std::unordered_map<int64_t, std::list<CacheEntry>> cache_entry_map;
  // Total cache entries across all compile scopes.
  size_t total_cache_entry_count{0};
  // Entries removed from their lists while `pinned` is non-zero. Raw
  // CacheEntry* and PrecompileEntry* reach Python only under a CachePin (the
  // compile callback's argument and the lists it reads), so removed nodes are
  // parked here, unreachable, and freed once the last pin drops.
  std::list<CacheEntry> graveyard;
  std::list<PrecompileEntry> precompile_graveyard;
  size_t pinned{0};
  // Frame state to detect dynamic shape dims in the default compile scope.
  py::dict frame_state;
  // Isolated compile scopes must not teach the default scope which dimensions
  // to generalize.
  std::unordered_map<int64_t, py::dict> region_frame_state_map;
  std::mutex region_frame_state_mutex;
  // Actions to apply to all frames with this code object (non-isolated).
  // Read on every intercepted frame, so the mutex guarding it is per-ExtraState
  // rather than process-wide: two threads running different functions must not
  // serialize against each other here.
  FrameExecStrategy strategy{DEFAULT, DEFAULT};
  std::mutex strategy_mutex;
  // Monotonic token for the last global strategy write. Tokens come from a
  // process-wide counter, so resetting a code object's ExtraState cannot make
  // a stale owner appear current again.
  uint64_t strategy_generation{0};
  // Per-region strategies for isolated compiles. When an isolated region
  // hits its recompile limit, only that region goes RUN_ONLY.
  std::unordered_map<int64_t, FrameExecStrategy> region_strategy_map;

  ExtraState();
  ExtraState(const ExtraState&) = delete;
  ExtraState& operator=(const ExtraState&) = delete;
  ExtraState(ExtraState&&) = delete;
  ExtraState& operator=(ExtraState&&) = delete;
  ~ExtraState();
  std::list<CacheEntry>& cache_entry_list(int64_t isolate_recompiles_id);
  bool has_relevant_entries(int64_t isolate_recompiles_id) const;
  // Callers hold cache_mutex. Entries are found by the IDENTITY of the guard
  // manager that owns them, never by address: a std::list node is recycled, so
  // a fresh entry at an old address would otherwise pass for the old one.
  CacheEntry* find_entry(
      int64_t isolate_recompiles_id,
      PyObject* guard_manager);
  CacheEntry* find_entry(PyObject* guard_manager);
  void move_to_front(CacheEntry* cache_entry);
  void move_to_back(CacheEntry* cache_entry);
  // live_guard_manager is the wrapper that OWNS cache_entry (CacheEntry's own
  // guard_manager). It is what establishes, under the lock, that the raw
  // cache_entry read before the lock is still the entry it was.
  void invalidate(
      CacheEntry* cache_entry,
      py::object deleted_guard_manager,
      py::object live_guard_manager);
} ExtraState;

using ExtraStateRef = std::shared_ptr<ExtraState>;

// Acquiring a mutex while holding the GIL deadlocks against a thread that holds
// the mutex and needs the GIL. Nothing under cache_mutex runs Python or drops
// the GIL (see ExtraState), so contention should not arise; if it does anyway,
// take the uncontended fast path without touching the GIL and release it before
// blocking so the owner can finish rather than deadlock.
class VISIBILITY_HIDDEN CacheLock {
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

// What guard_manager.extra_state holds on the Python side. Weak on purpose: the
// entries own their guard managers, so a strong reference here would close a
// cycle Python's GC cannot see. invalidate() is a no-op once the state is gone.
struct VISIBILITY_HIDDEN ExtraStateHandle {
  std::weak_ptr<ExtraState> state;

  void invalidate(
      CacheEntry* cache_entry,
      py::object deleted_guard_manager,
      py::object live_guard_manager);
};

// Keeps every CacheEntry and PrecompileEntry handed to Python allocated (see
// ExtraState::graveyard) for as long as the pin lives.
class VISIBILITY_HIDDEN CachePin {
 public:
  explicit CachePin(ExtraStateRef state);
  CachePin(const CachePin&) = delete;
  CachePin& operator=(const CachePin&) = delete;
  CachePin(CachePin&&) = delete;
  CachePin& operator=(CachePin&&) = delete;
  ~CachePin();

 private:
  ExtraStateRef state_;
};

// Owned copies taken under cache_mutex. A CacheEntry* or the c_str() of its
// trace_annotation must not outlive the lock: unload or invalidate() on another
// thread can free either the moment it is released.
struct VISIBILITY_HIDDEN LookupResult {
  // null: guard evaluation raised; None: miss; otherwise the code to run.
  py::object code;
  std::string trace_annotation;
};

#else

typedef struct ExtraState ExtraState;
typedef struct PrecompileEntry PrecompileEntry;

#endif

// Helper to extract the first cache_entry for a given isolate_recompiles scope.
// Ownership contract
// args
//  - extra_state: Borrowed
//  - isolate_recompiles_id: The scope to extract from (-1 = default)
// return
//  - CacheEntry: Borrowed; valid only while the caller holds a CachePin.
CacheEntry* extract_cache_entry(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id);

// Returns either the previously stored frame state for this compile scope or an
// empty dict.
// Ownership contract
// args
//  - extra_state: Borrowed
//  - isolate_recompiles_id: Compile scope (-1 = default)
// return
//  - frame state: New reference.
FrameState* extract_frame_state(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id);

// Returns the FrameExecStrategy stored in extra_state.
// Ownership contract
// args
//  - extra_state: Borrowed
FrameExecStrategy extra_state_get_exec_strategy(ExtraState* extra_state);

uint64_t extra_state_get_exec_strategy_token(
    ExtraState* extra_state,
    FrameExecStrategy* strategy);

uint64_t extra_state_set_exec_strategy_with_token(
    ExtraState* extra_state,
    FrameExecStrategy strategy,
    FrameExecStrategy* prior_strategy);

bool extra_state_compare_and_set_exec_strategy(
    ExtraState* extra_state,
    uint64_t expected_generation,
    FrameExecStrategy strategy);

// Set the FrameExecStrategy to be done to all frames with code object
// corresponding to this extra_state. Ownership contract
// - extra_state: Borrowed
void extra_state_set_exec_strategy(
    ExtraState* extra_state,
    FrameExecStrategy strategy);

// Get the exec strategy for a specific isolate_recompiles region.
// Falls back to the global strategy if no per-region strategy is set.
FrameExecStrategy extra_state_get_region_exec_strategy(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id);

// Set the exec strategy for a specific isolate_recompiles region.
void extra_state_set_region_exec_strategy(
    ExtraState* extra_state,
    int64_t isolate_recompiles_id,
    FrameExecStrategy strategy);

// This is passed as freefunc to _PyEval_RequestCodeExtraIndex: CPython calls it
// from _PyCode_SetExtra when a slot is overwritten and from code_dealloc. It
// drops the code object's reference to the ExtraState; the state itself is
// freed when the last lookup or pin holding it lets go. On a slot that
// reset_extra_state is already tearing down it returns without freeing; see
// ExtraStateHolder in extra_state.cpp.

// Developer note - You should not call this function directly. CPython does.
// If you are in a situation trying to call this function, consider if
// reset_extra_state should be called.
void destroy_extra_state(void* obj);

// Detaches the ExtraState from the code object. Safe on a code object that has
// none or whose state is already being torn down. A state that Python run by
// the teardown wrote to this same code object (see init_and_set_extra_state)
// is installed once the old one is gone, unless a later reset issued by that
// same teardown asked to forget it: the last writer wins.
void reset_extra_state(PyCodeObject* code);

// Extracts the backend fn from the callback.
PyObject* get_backend(PyObject* callback);
// Turn on the cache-key lookup inside get_backend. Called once, when the first
// backend carrying one is constructed; see cache_entry.cpp.
void enable_precompile_cache_keys();

#ifdef __cplusplus

} // extern "C"

// Ownership contract
// args
//  - code: Borrowed
// return
//  - extra_state: a reference, or nullptr when the code object has none.
ExtraStateRef get_extra_state(PyCodeObject* code);

// Returns the code object's extra state, creating and installing one when
// get_extra_state returned nullptr. While the previous state is still being
// torn down the slot is left alone: the new state is parked on the holder,
// served by get_extra_state to every later caller in that teardown, and
// installed by reset_extra_state once the slot is clear. Allocating the state
// can run Python (see the definition), so a state another writer installed or
// parked meanwhile is returned instead of the fresh one.
ExtraStateRef init_and_set_extra_state(PyCodeObject* code);

// Lookup the cache held by extra_state. Guards run with cache_mutex released.
// Ownership contract
// args
//  - extra_state: Borrowed
// return:
//   - result->code: null if guard evaluation raised, None on a miss.
void lookup(
    ExtraState* extra_state,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    LookupResult* result,
    bool is_skip_guard_eval_unsafe);

// Try to resolve a cache lookup without materializing frame locals or running
// guard managers. Returns true when the lookup is complete (hit or miss), and
// false when the caller must fall back to lookup().
bool try_lookup_without_guard_eval(
    ExtraState* extra_state,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    LookupResult* result,
    bool is_skip_guard_eval_unsafe);

// Create a new cache entry at extra_state holding on to guarded_code.
// Ownership contract
// args
//  - extra_state: Borrowed
//  - guarded_code: Borrowed
// return:
//  - the new entry's code and trace annotation
LookupResult create_cache_entry(
    const ExtraStateRef& extra_state,
    PyObject* guarded_code,
    PyObject* callback);

// Returns the list of CacheEntry corresponding to code_obj.
// Warning: the entries are owned by C++ and stay valid only while the calling
// compile callback (a CachePin) is in flight, or until the next clear or reset.
py::list _debug_get_cache_entry_list(const py::handle& code_obj);
// Returns the list of CacheEntry for a given isolate_recompiles_id bucket.
// Same lifetime warning as _debug_get_cache_entry_list.
py::list _get_cache_entries_for_region(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id);
void _clear_cache_entries_for_region(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id);
size_t _get_total_cache_entry_count(const py::handle& code_obj);
void _reset_precompile_entries(const py::handle& code_obj);
void _reset_precompile_entries_for_owner(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id,
    const py::handle& owner);
void _reset_precompile_entries_for_region(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id);
void _load_precompile_entry(
    const py::handle& code_obj,
    py::object guard_manager,
    py::object dynamo_code,
    int64_t isolate_recompiles_id,
    py::object owner);
py::list _debug_get_precompile_entries(const py::handle& code_obj);
bool _has_precompile_entries(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id);
void _set_lru_cache(const py::object& boolean);

#endif

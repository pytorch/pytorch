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
#include <atomic>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

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

typedef struct VISIBILITY_HIDDEN ExtraState {
  // A pointer to the orig_code object to prevent race conditions in invalidate
  // function.
  PyCodeObject* orig_code;
  std::list<PrecompileEntry> precompile_entries;
  // Per-compile cache map: isolate_recompiles_id -> list of CacheEntry.
  // id -1 is the default (non-isolated) bucket. id >= 0 are isolated compiles.
  // All cache entries live in this map — there is no separate default list.
  std::unordered_map<int64_t, std::list<CacheEntry>> cache_entry_map;
  // Total cache entries across all compile scopes (for O(1)
  // _get_total_cache_entry_count)
  size_t total_cache_entry_count{0};
  // Lock ordering: a thread that needs both convert_frame.compile_lock and
  // this cache_mutex must take compile_lock FIRST (reset()/remove_from_cache
  // do). cache_mutex is recursive and drops the GIL while waiting; taking it
  // before compile_lock anywhere would risk a cycle against that path.
  mutable std::recursive_mutex cache_mutex;
  // Frame state to detect dynamic shape dims in the default compile scope.
  py::dict frame_state;
  // Per-region frame_state, so an isolated compile's frame_id numbers
  // independently of the default scope.
  std::unordered_map<int64_t, py::dict> region_frame_state_map;
  // Guards frame_state and region_frame_state_map alike: the module runs
  // without the GIL on free-threaded builds, so the default dict's move in
  // clear_in_place and its read in extract_frame_state need the same exclusion
  // the region map already has.
  std::mutex region_frame_state_mutex;
  // Actions to apply to all frames with this code object (non-isolated).
  // Read on every intercepted frame, so the default-region read is a lock-free
  // atomic load: an 8-byte trivially-copyable struct, lock-free on every
  // platform PyTorch builds for (static_assert below). Written only under
  // strategy_mutex, which also covers strategy_generation and
  // region_strategy_map, so compare_and_set still moves strategy and
  // generation together.
  std::atomic<FrameExecStrategy> strategy{FrameExecStrategy{DEFAULT, DEFAULT}};
  static_assert(std::atomic<FrameExecStrategy>::is_always_lock_free);
  std::mutex strategy_mutex;
  // Monotonic token for the last global strategy write. Tokens come from a
  // process-wide counter, so resetting a code object's ExtraState cannot make
  // a stale owner appear current again.
  uint64_t strategy_generation{0};
  // Per-region strategies for isolated compiles. When an isolated region
  // hits its recompile limit, only that region goes RUN_ONLY.
  std::unordered_map<int64_t, FrameExecStrategy> region_strategy_map;
  // Invalidations that arrived while cache_mutex was contended. invalidate()
  // must never BLOCK on cache_mutex: it is reached from weakref.finalize,
  // which GC can fire while ANOTHER ExtraState's cache_mutex is held during
  // its guard evaluation, and two threads doing that against each other's
  // states deadlock (CacheLock releases only the GIL, not the peer's lock).
  // Parked requests are applied by the next holder of cache_mutex.
  std::mutex pending_invalidation_mutex;
  std::vector<std::pair<py::object, py::object>> pending_invalidations;
  // Cheap early-out for drain_pending_invalidations, so the hot lookup paths
  // do not take pending_invalidation_mutex when nothing is parked.
  std::atomic<bool> has_pending_invalidations{false};
  // Cache evictions that could not run where they were requested. Two ways
  // that happens: an owner-scoped uninstall reached from weakref.finalize (a
  // dead CompilePackage) must never BLOCK on cache_mutex, for exactly the
  // ABBA reason invalidate() must not; and ANY eviction arriving on a thread
  // whose own lookup is mid-guard-evaluation -- the recursive cache_mutex
  // admits it -- must not free or relink nodes that lookup is iterating
  // (reset_code from Python run BY A GUARD is the same-thread use-after-free
  // this closes). Guarded by pending_invalidation_mutex; applied by the next
  // cache_mutex holder whose cache_python_depth is zero. Known limit: an
  // eviction parked from inside a lookup is applied AFTER entries that same
  // Python then installs or compiles, and takes them too unless (as install()
  // does with a fresh owner token) they are keyed apart from it.
  struct PendingEviction {
    enum Kind : uint8_t {
      CLEAR_ALL, // every cache and precompile entry
      OWNER, // precompile entries of (region_id, owner)
      PRECOMPILE_REGION, // precompile entries of region_id
      PRECOMPILE_ALL, // every precompile entry
      CACHE_REGION, // cache entries of region_id
    };
    Kind kind;
    int64_t region_id;
    py::object owner;
  };
  std::vector<PendingEviction> pending_evictions;
  std::atomic<bool> has_pending_evictions{false};
  // Count of live lookup() snapshots iterating raw entry pointers with
  // cache_mutex released for guard evaluation. lookup() raises it under the
  // lock at depth 0, then drops the lock; another thread that takes cache_mutex
  // sees a non-zero count and parks every destroy/relink
  // (apply_pending_evictions, drain_pending_invalidations, invalidate,
  // clear_in_place) so those snapshots stay valid. It is thus a cross-thread
  // signal decremented on lookup()'s return paths after the lock is dropped, so
  // it is atomic rather than cache_mutex-guarded.
  std::atomic<size_t> cache_python_depth{0};

  ExtraState(PyCodeObject* orig_code_arg);
  std::list<CacheEntry>& cache_entry_list(int64_t isolate_recompiles_id);
  bool has_relevant_entries(int64_t isolate_recompiles_id);
  void move_to_front(CacheEntry* cache_entry, std::list<CacheEntry>& entries);
  void move_to_back(CacheEntry* cache_entry);
  // live_guard_manager is the wrapper that OWNS the entry to invalidate
  // (CacheEntry's own guard_manager); the entry is re-located by its identity
  // under the lock, never by an address read before the lock.
  void invalidate(
      py::object deleted_guard_manager,
      py::object live_guard_manager);
  // The identity-search body of invalidate. Caller must hold cache_mutex at
  // cache_python_depth zero: this relinks the entry's list.
  void invalidate_locked(
      const py::object& deleted_guard_manager,
      const py::object& live_guard_manager);
  // Apply invalidations parked while cache_mutex was contended. No-op unless
  // cache_python_depth is zero, as for apply_pending_evictions. Caller must
  // hold cache_mutex (and must NOT hold pending_invalidation_mutex).
  void drain_pending_invalidations();
  // Park an eviction for the next depth-zero cache_mutex holder.
  void park_eviction(PendingEviction eviction);
  // Apply parked evictions, moving the evicted nodes into the caller's
  // containers, which the caller destroys AFTER cache_mutex releases. No-op
  // unless cache_python_depth is zero: an in-flight same-thread lookup's
  // iterators must see its lists untouched until it finishes. dead_evictions
  // receives the drained PendingEviction records so their owner py::objects
  // are destroyed by the caller after the lock too, never under it. Caller
  // must hold cache_mutex (and must NOT hold pending_invalidation_mutex).
  void apply_pending_evictions(
      std::list<PrecompileEntry>& dead_precompile,
      std::unordered_map<int64_t, std::list<CacheEntry>>& dead_cache,
      std::vector<PendingEviction>& dead_evictions);
  // Empty this state back to freshly-constructed contents WITHOUT freeing it.
  // reset_code uses this instead of destroy: destroying while another thread
  // is blocked on cache_mutex (CacheLock releases the GIL while it waits)
  // would delete the very mutex the waiter is parked on.
  void clear_in_place();
} ExtraState;

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
//  - CacheEntry: Borrowed.
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

// Ownership contract
// args
//  - code: Borrowed
// return
//  - extra_state: Borrowed.
ExtraState* get_extra_state(PyCodeObject* code);

// This is passed as freefunc to _PyEval_RequestCodeExtraIndex. This acts as a
// deleter for the object on extra scratch space. This function is called
// internally in _PyCode_SetExtra and also during the code deallocation.

// Destroys the extra state by deleting cache_entry, frame state and finally
// freeing the constructed extra state.

// Developer note - You should not call this function directly. This is called
// directly inside set_extra_state. If you are in a situation trying to call
// this function, consider if set_extra_state should be called.
void destroy_extra_state(void* obj);

// Empties the code object's ExtraState in place, without freeing it. This is
// what torch._dynamo.reset_code must use: destroying the state (via
// set_extra_state(code, NULL)) while another thread is blocked on its
// cache_mutex -- CacheLock releases the GIL while waiting, so reset can run
// concurrently -- deletes the mutex under the waiter. The husk stays attached
// to the still-alive code object and behaves like a fresh state; the real
// destroy happens only from the code object's dealloc, when no frame of that
// code can be mid-lookup. Callers racing an in-flight COMPILE (which holds a
// Python-side snapshot of this code's cache entries) must additionally hold
// convert_frame.compile_lock, as torch._dynamo.reset() and remove_from_cache
// do; this function only makes the reset safe against concurrent lookups.
// Ownership contract
// args
//  - code: Borrowed
void reset_extra_state(PyCodeObject* code);

// Clears the existing object sitting on the extra scratch spance and sets it
// up with the new state. Note that _PyCode_SetExtra calls the
// destroy_extra_state deleter internally, and therefore we don't call it
// explicitly here.

// Ownership contract
// args
//  - extra_state: Stolen
// return
//  - there is no return, but the extra_state is stolen, so it becomes
//  set_extra_state responsibility to clean it up. It will be deleted during
//  the reset_code, when the set_extra_state is called with NULL.

// Invariant - Don't set the extra state for the extra state that is already on
// the code object. Otherwise, we will first free up the old extra state
// (which is also the new extra state) and write something invalid on the
// scratch space.
void set_extra_state(PyCodeObject* code, ExtraState* extra_state);

// Creates a new extra state and put it on the extra scratch space of the code
// object.

// Ownership contract
// args
//  - code: Borrowed
// return:
//   - extra_state: New reference.
// These references are then further passed to set_extra_state which becomes
// the final owner of these references.
ExtraState* init_and_set_extra_state(PyCodeObject* code);

// Turn on the cache-key lookup inside get_backend. Called once, when the first
// backend carrying one is constructed; see cache_entry.cpp.
void enable_precompile_cache_keys();

#ifdef __cplusplus

} // extern "C"

// Extracts the backend fn from the callback. Returns an OWNED reference; lives
// outside the extern "C" block because it returns a py::object. Only called
// from C++ (cache_entry.cpp, the frame evaluator).
py::object get_backend(PyObject* callback);

// Attribute lookup that returns an owned reference on success and an empty
// py::object when absent, WITHOUT raising; name must be an interned str.
// Defined in cache_entry.cpp; use this instead of py::hasattr on hot frame
// paths.
py::object lookup_optional(py::handle handle, PyObject* name);

// Create a new cache entry at extra_state holding on to guarded_code. Only
// called from C++ (the frame evaluator), so it lives outside the extern "C"
// block: the new entry's code (owned) and trace annotation are filled into the
// caller's py::object / std::string under the cache lock, and the returned
// pointer must not be dereferenced after this returns -- a concurrent clear
// can destroy the entry the moment the lock drops.
// Ownership contract
// args
//  - extra_state: Borrowed
//  - guarded_code: Borrowed
// return:
//  - cache_entry: Borrowed reference
CacheEntry* create_cache_entry(
    ExtraState* extra_state,
    PyObject* guraded_code,
    PyObject* callback,
    py::object* code_out,
    std::string* trace_annotation_out);

// Lookup the cache held by extra_state. Only called from C++ (the frame
// evaluator), so it lives outside the extern "C" block: trace_annotation is
// copied out under the cache lock into the caller's std::string, because the
// entry that owns the original may be destroyed as soon as this returns. The
// code object is handed back as a NEW reference for the same reason: the entry
// that owns it can be evicted the moment the cache lock drops.
// Ownership contract
// args
//  - extra_state: Borrowed
// return:
//   - Py_None or PyCodeObject: New reference; caller owns. nullptr on a guard
//     error.
void lookup(
    ExtraState* extra_state,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    PyObject** maybe_cached_code,
    std::string* trace_annotation,
    bool is_skip_guard_eval_unsafe);

// Try to resolve a cache lookup without materializing frame locals or running
// guard managers. Returns true when the lookup is complete (hit or miss), and
// false when the caller must fall back to lookup(). On true, *maybe_cached_code
// is a NEW reference the caller owns (as for lookup()); on false, untouched.
bool try_lookup_without_guard_eval(
    ExtraState* extra_state,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    PyObject** maybe_cached_code,
    std::string* trace_annotation,
    bool is_skip_guard_eval_unsafe);

// Returns the list of CacheEntry corresponding to code_obj.
// Warning: returns references whose lifetimes are controlled by C++
py::list _debug_get_cache_entry_list(const py::handle& code_obj);
// Returns the list of CacheEntry for a given isolate_recompiles_id bucket.
// Warning: returns references whose lifetimes are controlled by C++
py::list _get_cache_entries_for_region(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id);
size_t _get_cache_entry_count_for_region(
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

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
#include <unordered_set>
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

  PrecompileEntry(py::object gm, py::object c, int64_t region_id);
} PrecompileEntry;

struct VISIBILITY_HIDDEN PrecompileEntrySnapshot {
  py::object guard_manager;
  int64_t isolate_recompiles_id;

  explicit PrecompileEntrySnapshot(const PrecompileEntry& entry);
};

class CacheEntryHandle;
struct CacheEntrySnapshot;
using CacheEntryPtr = std::shared_ptr<CacheEntry>;
using CacheEntryList = std::list<CacheEntryPtr>;
using PrecompileEntryPtr = std::shared_ptr<PrecompileEntry>;
using PrecompileEntryList = std::list<PrecompileEntryPtr>;

typedef struct VISIBILITY_HIDDEN ExtraState {
  // A pointer to the orig_code object to prevent race conditions in invalidate
  // function.
  PyCodeObject* orig_code;
  PrecompileEntryList precompile_entries;
  // Per-compile cache map: isolate_recompiles_id -> list of CacheEntry.
  // id -1 is the default (non-isolated) bucket. id >= 0 are isolated compiles.
  // All cache entries live in this map — there is no separate default list.
  std::unordered_map<int64_t, CacheEntryList> cache_entry_map;
  // Total cache entries across all compile scopes (for O(1)
  // has_any_cache_entries)
  size_t total_cache_entry_count{0};
  mutable std::recursive_mutex cache_mutex;
  size_t active_cache_lookups{0};
  bool pending_full_cache_reset{false};
  bool pending_full_precompile_reset{false};
  std::unordered_set<int64_t> pending_cache_region_clears;
  std::unordered_set<int64_t> pending_precompile_region_resets;
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

  ExtraState(PyCodeObject* orig_code_arg);
  CacheEntryList& cache_entry_list(int64_t isolate_recompiles_id);
  bool has_any_cache_entries() const;
  bool has_relevant_entries(int64_t isolate_recompiles_id) const;
  void move_to_front(CacheEntry* cache_entry, CacheEntryList& entries);
  void move_to_back(CacheEntry* cache_entry);
  void reset();
  // live_guard_manager identifies the cache entry independently of the weak
  // handle, which may expire while the caller waits for cache_mutex.
  void invalidate(const CacheEntryHandle& cache_entry, py::object deleted_guard_manager, py::object live_guard_manager);
  bool has_pending_destructive_cache_mutation() const;
  void apply_pending_cache_mutations(std::vector<CacheEntryPtr>& retired_cache_entries,
                                     std::vector<PrecompileEntryPtr>& retired_precompile_entries);
  void clear_cache_entries(std::vector<CacheEntryPtr>& retired_cache_entries);
  void clear_cache_entries_for_region(int64_t isolate_recompiles_id, std::vector<CacheEntryPtr>& retired_cache_entries);
  void clear_precompile_entries(std::vector<PrecompileEntryPtr>& retired_precompile_entries);
  void clear_precompile_entries_for_region(int64_t isolate_recompiles_id,
                                           std::vector<PrecompileEntryPtr>& retired_precompile_entries);
} ExtraState;

#else

typedef struct ExtraState ExtraState;
typedef struct PrecompileEntry PrecompileEntry;

#endif

// Returns either the previously stored frame state for this compile scope or an
// empty dict.
// Ownership contract
// args
//  - extra_state: Borrowed
//  - isolate_recompiles_id: Compile scope (-1 = default)
// return
//  - frame state: New reference.
FrameState* extract_frame_state(ExtraState* extra_state, int64_t isolate_recompiles_id);

// Returns the FrameExecStrategy stored in extra_state.
// Ownership contract
// args
//  - extra_state: Borrowed
FrameExecStrategy extra_state_get_exec_strategy(ExtraState* extra_state);

uint64_t extra_state_get_exec_strategy_token(ExtraState* extra_state, FrameExecStrategy* strategy);

uint64_t extra_state_set_exec_strategy_with_token(ExtraState* extra_state,
                                                  FrameExecStrategy strategy,
                                                  FrameExecStrategy* prior_strategy);

bool extra_state_compare_and_set_exec_strategy(ExtraState* extra_state,
                                               uint64_t expected_generation,
                                               FrameExecStrategy strategy);

// Set the FrameExecStrategy to be done to all frames with code object
// corresponding to this extra_state. Ownership contract
// - extra_state: Borrowed
void extra_state_set_exec_strategy(ExtraState* extra_state, FrameExecStrategy strategy);

// Get the exec strategy for a specific isolate_recompiles region.
// Falls back to the global strategy if no per-region strategy is set.
FrameExecStrategy extra_state_get_region_exec_strategy(ExtraState* extra_state, int64_t isolate_recompiles_id);

// Set the exec strategy for a specific isolate_recompiles region.
void extra_state_set_region_exec_strategy(ExtraState* extra_state,
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

// Clear a code object's existing state in place. Keeping the allocation alive
// lets an in-flight cache lookup finish after CacheLock temporarily releases the
// GIL while waiting or evaluating a Python guard.
void reset_extra_state(PyCodeObject* code);

// Returns the existing extra state, or creates it exactly once for this code
// object. Safe when multiple free-threaded callers race on a cold code object.

// Ownership contract
// args
//  - code: Borrowed
// return:
//   - extra_state: New reference.
// These references are then further passed to set_extra_state which becomes
// the final owner of these references.
ExtraState* init_and_set_extra_state(PyCodeObject* code);

// Extracts the backend fn from the callback.
PyObject* get_backend(PyObject* callback);

#ifdef __cplusplus

} // extern "C"

struct VISIBILITY_HIDDEN CacheLookupResult {
  // Null signals a guard error; Py_None signals a cache miss. A hit owns the
  // code reference and annotation independently of the cache entry.
  py::object code;
  std::string trace_annotation;
};

py::object extract_cache_entry_snapshot(ExtraState* extra_state, int64_t isolate_recompiles_id);

CacheEntrySnapshot create_cache_entry(ExtraState* extra_state, PyObject* guarded_code, PyObject* callback);

void lookup(ExtraState* extra_state,
            FrameLocalsMapping* f_locals,
            PyObject* backend,
            int64_t isolate_recompiles_id,
            CacheLookupResult* result,
            bool is_skip_guard_eval_unsafe);

// Try to resolve a cache lookup without materializing frame locals or running
// guard managers. Returns true when the lookup is complete (hit or miss), and
// false when the caller must fall back to lookup().
bool try_lookup_without_guard_eval(ExtraState* extra_state,
                                   PyObject* backend,
                                   int64_t isolate_recompiles_id,
                                   CacheLookupResult* result,
                                   bool is_skip_guard_eval_unsafe);

// Returns owning snapshots of the CacheEntry objects corresponding to code_obj.
py::list _debug_get_cache_entry_list(const py::handle& code_obj);
// Returns owning snapshots for a given isolate_recompiles_id bucket.
py::list _get_cache_entries_for_region(const py::handle& code_obj, int64_t isolate_recompiles_id);
void _clear_cache_entries_for_region(const py::handle& code_obj, int64_t isolate_recompiles_id);
size_t _get_total_cache_entry_count(const py::handle& code_obj);
void _reset_precompile_entries(const py::handle& code_obj);
void _reset_precompile_entries_for_region(const py::handle& code_obj, int64_t isolate_recompiles_id);
void _load_precompile_entry(const py::handle& code_obj,
                            py::object guard_manager,
                            py::object dynamo_code,
                            int64_t isolate_recompiles_id);
py::list _debug_get_precompile_entries(const py::handle& code_obj);
void _set_lru_cache(const py::object& boolean);

#endif

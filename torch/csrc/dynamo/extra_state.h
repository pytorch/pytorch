#pragma once

#include <Python.h>

#include <torch/csrc/dynamo/framelocals_mapping.h>

#ifdef __cplusplus

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>
#include <list>
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
//
// Everything that reads or writes an ExtraState does so inside
// extra_state.cpp.  An ExtraState* never leaves that file except through the
// guard manager's back-references, so the API below is keyed on the code
// object.

#ifdef __cplusplus

typedef struct VISIBILITY_HIDDEN PrecompileEntry {
  py::object guard_manager;
  py::object code;
  void* root_mgr;

  PrecompileEntry(py::object gm, py::object c);
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
  // has_any_cache_entries)
  size_t total_cache_entry_count{0};
  // Frame state to detect dynamic shape dims
  py::dict frame_state;
  // Actions to apply to all frames with this code object (non-isolated)
  FrameExecStrategy strategy{DEFAULT, DEFAULT};
  // Per-region strategies for isolated compiles. When an isolated region
  // hits its recompile limit, only that region goes RUN_ONLY.
  std::unordered_map<int64_t, FrameExecStrategy> region_strategy_map;

  ExtraState(PyCodeObject* orig_code_arg);
  std::list<CacheEntry>& cache_entry_list(int64_t isolate_recompiles_id);
  bool has_any_cache_entries() const;
  void move_to_front(CacheEntry* cache_entry, std::list<CacheEntry>& entries);
  void move_to_back(CacheEntry* cache_entry);
  void invalidate(CacheEntry* cache_entry, py::object deleted_guard_manager);
} ExtraState;

#else

typedef struct ExtraState ExtraState;
typedef struct PrecompileEntry PrecompileEntry;

#endif

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

// Extracts the backend fn from the callback.
PyObject* get_backend(PyObject* callback);

#ifdef __cplusplus

} // extern "C"

// What the Dynamo callback is handed.  cache_entry is borrowed and is only
// valid until the code object's cache is reset.
struct CompileInputs {
  CacheEntry* cache_entry{nullptr};
  py::object frame_state;
};

// Returns false when the code object has no cache state and create is false.
// Otherwise installs a state if needed and returns the region's strategy:
// global SKIP actions override the region strategy, other global actions are
// not inherited.
bool get_frame_exec_strategy(
    PyCodeObject* code,
    int64_t isolate_recompiles_id,
    bool create,
    FrameExecStrategy* strategy);

// Sets the strategy for every frame of this code object.
void set_code_exec_strategy(PyCodeObject* code, FrameExecStrategy strategy);

// Try to resolve a cache lookup without materializing frame locals or running
// guard managers. Returns true when the lookup is complete (hit or miss), and
// false when the caller must fall back to lookup().
bool try_lookup_without_guard_eval(
    PyCodeObject* code,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    PyObject** maybe_cached_code,
    const char** trace_annotation,
    bool is_skip_guard_eval_unsafe);

// Lookup the cache held by a code object.
// Ownership contract
// args:
//   - code: Borrowed reference
//   - f_locals: Borrowed reference
//   - backend: Borrowed reference
// return:
//   - maybe_cached_code: Borrowed reference or Py_None
//   - trace_annotation: Borrowed pointer to cache entry
void lookup(
    PyCodeObject* code,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    int64_t isolate_recompiles_id,
    PyObject** maybe_cached_code,
    const char** trace_annotation,
    bool is_skip_guard_eval_unsafe);

// Whether the region or the default bucket has any entries, for
// guard_complete_hook.
bool has_relevant_cache_entries(
    PyCodeObject* code,
    int64_t isolate_recompiles_id);

CompileInputs get_compile_inputs(
    PyCodeObject* code,
    int64_t isolate_recompiles_id);

// Applies the callback's result: the new strategy when apply_to_code, and a
// cache entry for guarded_code unless it is None.  Returns the new entry, or
// null; it is borrowed and only valid until the code object's cache is reset.
CacheEntry* record_compile_result(
    PyCodeObject* code,
    int64_t isolate_recompiles_id,
    bool apply_to_code,
    FrameExecStrategy new_strategy,
    PyObject* guarded_code,
    PyObject* backend);

// Returns the list of CacheEntry corresponding to code_obj.
// Warning: returns references whose lifetimes are controlled by C++
py::list _debug_get_cache_entry_list(const py::handle& code_obj);
// Returns the list of CacheEntry for a given isolate_recompiles_id bucket.
// Warning: returns references whose lifetimes are controlled by C++
py::list _get_cache_entries_for_region(
    const py::handle& code_obj,
    int64_t isolate_recompiles_id);
size_t _get_total_cache_entry_count(const py::handle& code_obj);
void _reset_precompile_entries(const py::handle& code_obj);
void _load_precompile_entry(
    const py::handle& code_obj,
    py::object guard_manager,
    py::object dynamo_code);
py::list _debug_get_precompile_entries(const py::handle& code_obj);
bool _set_lru_cache(py::object boolean);

#endif

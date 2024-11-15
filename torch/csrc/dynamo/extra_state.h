#pragma once

#include <Python.h>

#ifdef __cplusplus

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>
#include <list>

namespace py = pybind11;

extern "C" {

#else

#include <stdbool.h>

#endif

// Flag to just run a frame normally
#define SKIP_CODE ((void*)0x1)
// Flag to run a frame and any recursive calls normally
#define SKIP_CODE_RECURSIVE ((void*)0x2)

// Points to the extra scratch space on the code object
extern Py_ssize_t extra_index;

// function to call when cache lookup errors
extern PyObject* guard_error_hook;

typedef PyObject FrameState;
typedef struct CacheEntry CacheEntry;

// ExtraState encasulates CacheEntry and FrameState. ExtraState is the highest
// level of abstraction of what is stored on the extra code object. Previously,
// we saved different parts on different extra indexes.  We prefer this way
// because of cleaner abstraction and faster SetExtra access.

#ifdef __cplusplus

typedef struct VISIBILITY_HIDDEN ExtraState {
  // List of cache entries for compiled code objects
  std::list<CacheEntry> cache_entry_list;
  // Frame state to detect dynamic shape dims
  py::dict frame_state;
  bool cache_limit_hit{false};

  CacheEntry* get_first_entry();
  void move_to_front(CacheEntry* cache_entry);
  void move_to_back(CacheEntry* cache_entry);
  void invalidate(CacheEntry* cache_entry, py::object deleted_guard_manager);
} ExtraState;

#else

typedef struct ExtraState ExtraState;

#endif

// Helper to extra the cache_entry from the extra state.
// Ownership contract
// args
//  - extra_state: Borrowed
// return
//  - CacheEntry: Borrowed.
CacheEntry* extract_cache_entry(ExtraState* extra_state);

// Returns either the previously stored frame state or an empty dict.
// Ownership contract
// args
//  - extra_state: Borrowed
// return
//  - extra_state->frame_state: Borrowed.
FrameState* extract_frame_state(ExtraState* extra_state);

// Returns if this extra_state is marked as cache limit hit.
// Ownership contract
// args
//  - extra_state: Borrowed
bool extra_state_cache_limit_hit(ExtraState* extra_state);

// Mark that extra_state has hit its cache limit hit.
// Ownership contract
// args
//  - extra_state: Borrowed
void set_extra_state_cache_limit_hit(ExtraState* extra_state, bool value);

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
// explicity here.

// Ownership contract
// args
//  - extra_state: Stolen
// return
//  - there is no return, but the extra_state is stolen, so it becomes
//  set_extra_state responsibility to clean it up. It will be deleted during
//  the reset_code/skip, when the set_extra_state is called with
//  NULL/SKIP_CODE/SKIP_CODE_RECURSIVE.

// Invariant - Dont set the extra state for the extra state that is already on
// the code object. Otherwise, we will first free up the old extra state
// (which is also the new extra state) and write something invalid on the
// scratch space.
void set_extra_state(PyCodeObject* code, ExtraState* extra_state);

// Creates a new extra state and put it on the extra scrach space of the code
// object.

// Ownership contract
// args
//  - code: Borrowed
// return:
//   - extra_state: New reference.
// These references are then further passed to set_extra_state which becomes
// the final owner of these references.
ExtraState* init_and_set_extra_state(PyCodeObject* code);

// Lookup the cache held by extra_state.
// Ownership contract
// args
//  - extra_state: Borrowed
//  - f_locals: Borrowed
// return:
//   - Py_None or PyCodeObject: Borrowed reference.
//   - Py_None or PyObject: Trace id of the compiled code.
void lookup(
    ExtraState* extra_state,
    PyObject* f_locals,
    PyObject* backend,
    PyObject** maybe_cached_code,
    const char** trace_annotation);

// Create a new cache entry at extra_state holding on to guarded_code.
// Ownership contract
// args
//  - extra_state: Borrowed
//  - guarded_code: Borrowed
// return:
//  - cache_entry: Borrowed reference
CacheEntry* create_cache_entry(
    ExtraState* extra_state,
    PyObject* guraded_code,
    PyObject* callback);

// Extracts the backend fn from the callback.
PyObject* get_backend(PyObject* callback);

#ifdef __cplusplus

} // extern "C"

// Returns the list of CacheEntry corresponding to code_obj.
// Warning: returns references whose lifetimes are controlled by C++
py::list _debug_get_cache_entry_list(const py::handle& code_obj);

#endif

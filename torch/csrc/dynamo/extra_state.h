#pragma once

#include <torch/csrc/utils/python_compat.h>

#ifdef __cplusplus
extern "C" {
#endif

// Flag to just run a frame normally
#define SKIP_CODE ((void*)0x1)

// Points to the extra scratch space on the code object
extern Py_ssize_t extra_index;

typedef PyObject FrameState;
typedef struct cache_entry CacheEntry;

// ExtraState encasulates CacheEntry and FrameState. ExtraState is the highest
// level of abstraction of what is stored on the extra code object. Previously,
// we saved different parts on different extra indexes.  We prefer this way
// because of cleaner abstraction and faster SetExtra access.

// TODO(anijain2305) - Consider making this a PyObject. Benefits are
//   1) Modular dealloc - destroy_extra_state just becomes Py_DECREF(extra)
//   2) We can directly send the extra object to convert_frame callback. One
//   data structure - easier to understand code.
// There might be some perf impact of going through a PyObject on the critical
// path, but it should not be too bad.
typedef struct {
  // Cache entry for the code object
  CacheEntry* cache_entry;
  // Frame state to detect dynamic shape dims
  FrameState* frame_state;
} ExtraState;


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
//  NULL/SKIP_CODE.

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

// get the cache entry out of a code object
PyObject* _debug_get_cache_entry_list(PyObject* self, PyObject* args);

#ifdef __cplusplus
} // extern "C"
#endif

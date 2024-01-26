#pragma once

#include <torch/csrc/utils/python_compat.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
Our cache resides on the extra scratch space of the code object. The structure
of the cache is as follows:

-> ExtraState
  -> CacheEntry
    -> check_fn
    -> optimized_code
    -> next
  -> FrameState

CacheEntry is a linked list, with each node containing the check_fn for guards
and the optimized code.

The frame_state is a PyDict that enables sharing between different frames. This
is used to detect dynamism in automatic dynamic shapes.

These two are encapsulated into a ExtraState.
*/

// Linked list of cache entries, where each cache entry stores
// the check_fn and the torch.compile optimized python bytecode.
typedef struct cache_entry {
  PyObject_HEAD
  // check the guards: lambda: <locals of user function>: bool
  PyObject* check_fn;
  // modified user bytecode (protected by check_fn's guards)
  PyCodeObject* code;
  // on a cache miss, linked list of next thing to try
  struct cache_entry* next;
} CacheEntry;

extern PyTypeObject CacheEntryType;

// Ownership contract
// args
//   - next: steals
//   - guarded_code: Borrowed
//  return
//   - CacheEntry*: new reference.
CacheEntry* create_cache_entry(
    CacheEntry* next,
    PyObject* guarded_code);

#ifdef __cplusplus
} // extern "C"
#endif

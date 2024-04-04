#pragma once

#include <Python.h>

#ifdef __cplusplus

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>
#include <list>

extern "C" {

#endif

/*
Our cache resides on the extra scratch space of the code object. The structure
of the cache is as follows:

-> ExtraState
  -> CacheEntry (list)
    -> check_fn
    -> code
  -> FrameState

CacheEntry is a linked list node containing the check_fn for guards
and the optimized code.

The FrameState is a PyDict that enables sharing between different frames. This
is used to detect dynamism in automatic dynamic shapes.

These two are encapsulated into a ExtraState.
*/

typedef struct CacheEntry CacheEntry;
typedef struct ExtraState ExtraState;

#ifdef __cplusplus

typedef struct VISIBILITY_HIDDEN CacheEntry {
  // check the guards: lambda: <locals of user function>: bool
  py::object check_fn;
  // modified user bytecode (protected by check_fn's guards)
  py::object code;
  // root guard manager if exists
  void* root_mgr{nullptr};
  // backend used to create this cache entry
  PyObject* backend{nullptr};
  // Reference to owning ExtraState
  ExtraState* _owner{nullptr};
  // Reference to this CacheEntry's location in owner's linked list
  std::list<CacheEntry>::iterator _owner_loc;

  CacheEntry(const py::handle& guarded_code, PyObject* backend);
  ~CacheEntry();

  // Warning: returns a reference whose lifetime is controlled by C++
  py::object next();
} CacheEntry;

#endif

// Returns borrowed reference
PyCodeObject* CacheEntry_get_code(CacheEntry* e);

// Returns a borrowed reference to CacheEntry as a PyObject
// Warning: lifetime is controlled by C++
PyObject* CacheEntry_to_obj(CacheEntry* e);

#ifdef __cplusplus
} // extern "C"
#endif

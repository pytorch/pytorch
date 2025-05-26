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
    -> guard_manager (a wrapper that contains the actual guard manager at its
attr named root)
    -> code
  -> FrameState

CacheEntry is a linked list node containing the guard_manager for guards
and the optimized code.

The FrameState is a PyDict that enables sharing between different frames. This
is used to detect dynamism in automatic dynamic shapes.

These two are encapsulated into a ExtraState.
*/

typedef struct CacheEntry CacheEntry;
typedef struct ExtraState ExtraState;

#ifdef __cplusplus

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED(
    "-Wdeprecated-copy-with-user-provided-dtor")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-copy-dtor")
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
typedef struct VISIBILITY_HIDDEN CacheEntry {
  // check the guards: lambda: <locals of user function>: bool
  py::object guard_manager;
  // modified user bytecode (protected by guard_manager's guards)
  py::object code;
  // CompileId corresponding to this compilation
  py::object compile_id;
  // root guard manager if exists
  void* root_mgr{nullptr};
  // diff guard root guard manager if exists
  void* diff_guard_root_mgr{nullptr};
  // backend used to create this cache entry
  PyObject* backend{nullptr};
  // Reference to owning ExtraState
  ExtraState* _owner{nullptr};
  // Reference to this CacheEntry's location in owner's linked list
  std::list<CacheEntry>::iterator _owner_loc;
  // Reference to string representation of the CompileContext
  std::string trace_annotation;

  CacheEntry(const py::handle& guarded_code, PyObject* backend);
  CacheEntry(const CacheEntry&) = default;
  CacheEntry(CacheEntry&&) = default;
  CacheEntry& operator=(const CacheEntry&) = default;
  CacheEntry& operator=(CacheEntry&&) = default;
  ~CacheEntry();

  // Warning: returns a reference whose lifetime is controlled by C++
  py::object next();

  void invalidate(py::object deleted_guard_manager);
  // Called from the python side to update the diff guard root manager
  void update_diff_guard_root_manager();
} CacheEntry;
C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()

#endif

// Returns borrowed reference
PyCodeObject* CacheEntry_get_code(CacheEntry* e);

// Returns borrowed string representation of CompileContext
const char* CacheEntry_get_trace_annotation(CacheEntry* e);

// Returns a borrowed reference to CacheEntry as a PyObject
// Warning: lifetime is controlled by C++
PyObject* CacheEntry_to_obj(CacheEntry* e);

#ifdef __cplusplus
} // extern "C"
#endif

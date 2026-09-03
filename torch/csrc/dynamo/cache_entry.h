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

These two are encapsulated into an ExtraState.
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
  // diff guard root guard manager if exists. guard_manager.diff_guard_root is
  // rebound on every recompile of the same code, so the entry keeps its own
  // reference to the object that owns the raw pointer; both are read under
  // cache_mutex (lookup copies them, the fast path inspects the pointer).
  py::object diff_guard_root;
  void* diff_guard_root_mgr{nullptr};
  // backend used to create this cache entry
  py::object backend;
  // Reference to owning ExtraState
  ExtraState* _owner{nullptr};
  // Reference to this CacheEntry's location in owner's linked list
  std::list<CacheEntry>::iterator _owner_loc;
  // The isolate_recompiles_id for this entry's bucket in cache_entry_map
  int64_t _isolate_recompiles_id{-1};
  // Reference to string representation of the CompileContext
  std::string trace_annotation;

  CacheEntry(const py::handle& guarded_code, PyObject* backend);
  CacheEntry(const CacheEntry&) = default;
  CacheEntry(CacheEntry&&) = default;
  CacheEntry& operator=(const CacheEntry&) = default;
  CacheEntry& operator=(CacheEntry&&) = default;
  ~CacheEntry();

  // The Python objects an invalidated entry gives up. Released by the caller
  // once cache_mutex is dropped: the decrefs and the attribute clears on
  // guard_manager run Python.
  struct Detached {
    py::object guard_manager;
    py::object code;
    py::object backend;
    py::object diff_guard_root;
  };
  // Runs no Python. Points this entry at deleted_guard_manager and hands back
  // what it held; the caller clears the old guard_manager's cache_entry and
  // extra_state attributes after unlocking.
  Detached invalidate(py::object deleted_guard_manager);
  // Called from the python side after guard_manager.diff_guard_root was
  // rebound. Reads the attribute before taking cache_mutex, swaps under it and
  // releases the old reference after it.
  void update_diff_guard_root_manager();
} CacheEntry;
C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()

} // extern "C"

// Borrowed on success, nullptr when absent, without raising. `name` must be an
// interned (or otherwise immortal) str.
PyObject* lookup_optional_attr(py::handle obj, PyObject* name);
#endif

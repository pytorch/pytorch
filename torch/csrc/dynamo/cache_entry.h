#pragma once

#include <Python.h>

#ifdef __cplusplus

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>
#include <list>
#include <vector>

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

typedef struct VISIBILITY_HIDDEN CacheEntry {
  // check the guards: lambda: <locals of user function>: bool
  py::object guard_manager;
  // modified user bytecode (protected by guard_manager's guards)
  py::object code;
  // CompileId corresponding to this compilation
  py::object compile_id;
  // root guard manager if exists
  void* root_mgr{nullptr};
  // owner of diff_guard_root_mgr
  py::object diff_guard_root;
  // diff guard root guard manager if exists
  void* diff_guard_root_mgr{nullptr};
  // backend used to create this cache entry
  py::object backend;
  // Reference to owning ExtraState; null for copies handed to Python
  ExtraState* _owner{nullptr};
  // Reference to this CacheEntry's location in owner's linked list
  std::list<CacheEntry>::iterator _owner_loc;
  // The isolate_recompiles_id for this entry's bucket in cache_entry_map
  int64_t _isolate_recompiles_id{-1};
  // Reference to string representation of the CompileContext
  std::string trace_annotation;
  // Assigned once per cached entry.  Copies of the entry and snapshots carry
  // their source's serial, so they can find the live entry again even if
  // another entry shares its guard manager.
  uint64_t serial{0};

  CacheEntry(const py::handle& guarded_code, PyObject* backend);
  CacheEntry(const CacheEntry&) = default;
  CacheEntry(CacheEntry&&) = default;
  CacheEntry& operator=(const CacheEntry&) = default;
  CacheEntry& operator=(CacheEntry&&) = default;

  // Leaves this entry matching nothing.  Dropping the references it held can
  // run Python, so they are moved into released for the caller to drop outside
  // the critical section.
  void invalidate(
      py::object deleted_guard_manager,
      std::vector<py::object>& released);
  // A copy that owns its references and does not belong to the cache.  The
  // non-owning pointers into the cache and guard managers are cleared.
  CacheEntry copy_for_python() const;

  // Implements __eq__ for Python.
  bool py_equal(const CacheEntry& other) const;
  // Implements __hash__ for Python.
  std::size_t py_hash() const;

} CacheEntry;

#endif

#ifdef __cplusplus
} // extern "C"
#endif

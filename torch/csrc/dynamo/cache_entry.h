#pragma once

#include <Python.h>

#ifdef __cplusplus

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>

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

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-copy-with-user-provided-dtor")
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
  py::object backend;
  // Reference to owning ExtraState
  ExtraState* _owner{nullptr};
  // Reference to this CacheEntry's location in owner's linked list
  std::list<std::shared_ptr<CacheEntry>>::iterator _owner_loc;
  // The isolate_recompiles_id for this entry's bucket in cache_entry_map
  int64_t _isolate_recompiles_id{-1};
  // Reference to string representation of the CompileContext
  std::string trace_annotation;
  uint64_t identity;
  // Serializes snapshotting and mutation through the weak Python handle.
  mutable std::recursive_mutex state_mutex;

  CacheEntry(const py::handle& guarded_code, PyObject* backend);
  CacheEntry(const CacheEntry&) = delete;
  CacheEntry(CacheEntry&&) = delete;
  CacheEntry& operator=(const CacheEntry&) = delete;
  CacheEntry& operator=(CacheEntry&&) = delete;
  ~CacheEntry();

  void invalidate(py::object deleted_guard_manager);
  // Called from the python side to update the diff guard root manager
  void update_diff_guard_root_manager();
} CacheEntry;

class VISIBILITY_HIDDEN CacheEntryHandle {
 public:
  explicit CacheEntryHandle(const std::shared_ptr<CacheEntry>& entry);

  std::shared_ptr<CacheEntry> lock() const;
  py::object backend() const;
  void update_diff_guard_root_manager() const;

 private:
  std::weak_ptr<CacheEntry> entry_;
};

struct VISIBILITY_HIDDEN CacheEntrySnapshot {
  py::object guard_manager;
  py::object code;
  py::object compile_id;
  py::object backend;
  int64_t isolate_recompiles_id;
  std::string trace_annotation;
  uint64_t identity;
  void* root_mgr;
  void* diff_guard_root_mgr;

  explicit CacheEntrySnapshot(const CacheEntry& entry);
};
C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()

#endif

// Returns borrowed reference
PyCodeObject* CacheEntry_get_code(CacheEntry* e);

// Returns borrowed string representation of CompileContext
const char* CacheEntry_get_trace_annotation(CacheEntry* e);

#ifdef __cplusplus
} // extern "C"
#endif

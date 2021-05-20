#pragma once

#include <functional>
#include <vector>
#include <torch/csrc/Types.h>

typedef std::function<void(PyObject*, PyObject*, bool)> THPCopyFunction;
struct THPCopyInfo {
  PyTypeObject* srcType;  // Python type of src tensor/storage
  THPCopyFunction copy;   // copy function
  bool non_blocking;             // true if copy implements an 'non_blocking' copy
  bool broadcast;         // true if the copy implements a broadcast copy
};
typedef std::vector<THPCopyInfo> THPCopyList;

inline bool tryTHPCopy(const THPCopyList& v, PyObject* dst, PyObject* src, bool non_blocking, bool broadcast)
{
  for (auto& i : v) {
    if (i.non_blocking == non_blocking && PyType_IsSubtype(Py_TYPE(src), i.srcType)) {
      (i.copy)(dst, src, broadcast);
      return true;
    }
  }
  return false;
}

inline bool THPCopy(const THPCopyList& v, PyObject* dst, PyObject* src, bool non_blocking, bool broadcast)
{
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (tryTHPCopy(v, dst, src, non_blocking, broadcast)) {
    return true;
  } else if (non_blocking && tryTHPCopy(v, dst, src, false, broadcast)) {
    return true;
  }
  THPUtils_setError("copy from %s to %s isn't implemented",
      THPUtils_typename(src), THPUtils_typename(dst));
  return false;
}

inline PyObject * THPStorageCopyMethod(const THPCopyList& v, PyObject *self, PyObject *args, PyObject *kwargs)
{
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  PyObject *src;
  int non_blocking = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,clang-diagnostic-writable-strings)
  static char *kwlist[] = {"source", "non_blocking", nullptr};
  // use int as parse type because bool not available in python2.
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i:copy_", kwlist, &src, &non_blocking)) {
    return nullptr;
  }

  if (!THPCopy(v, self, src, non_blocking, false)) {
    return nullptr;
  }

  Py_INCREF(self);
  return self;
}

template <typename THPStorageDst, typename THPStorageSrc, typename StorageDst, typename StorageSrc>
void THPInsertStorageCopyFunction(
  PyTypeObject *srcType,
  THPCopyList& copyList,
  void (*copyFunc)(LIBRARY_STATE_TYPE StorageDst* x, StorageSrc* z),
  bool non_blocking=false)
{
  auto wrapper = [copyFunc](PyObject* dst_, PyObject* src_, bool broadcast) {
    auto dst = ((THPStorageDst*)dst_)->cdata;
    auto src = ((THPStorageSrc*)src_)->cdata;

    PyThreadState *_save = nullptr;
    try {
      Py_UNBLOCK_THREADS;
      copyFunc(LIBRARY_STATE dst, src);
      Py_BLOCK_THREADS;
    } catch (...) {
      if (_save) {
        Py_BLOCK_THREADS;
      }
      throw;
    }
  };

  copyList.push_back({ srcType, wrapper, non_blocking, false });
}

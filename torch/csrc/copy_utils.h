#ifndef THP_COPY_UTILS_H
#define THP_COPY_UTILS_H

#include <functional>
#include <vector>
#include "Types.h"
#include "expand_utils.h"

typedef std::function<void(PyObject*, PyObject*, bool)> THPCopyFunction;
struct THPCopyInfo {
  PyTypeObject* srcType;  // Python type of src tensor/storage
  THPCopyFunction copy;   // copy function
  bool async;             // true if copy implements an 'async' copy
  bool broadcast;         // true if the copy implements a broadcast copy
};
typedef std::vector<THPCopyInfo> THPCopyList;

inline bool tryTHPCopy(const THPCopyList& v, PyObject* dst, PyObject* src, bool async, bool broadcast)
{
  for (auto it = v.begin(); it != v.end(); ++it) {
    if (it->async == async && PyType_IsSubtype(Py_TYPE(src), it->srcType)) {
      (it->copy)(dst, src, broadcast);
      return true;
    }
  }
  return false;
}

inline bool THPCopy(const THPCopyList& v, PyObject* dst, PyObject* src, bool async, bool broadcast)
{
  if (tryTHPCopy(v, dst, src, async, broadcast)) {
    return true;
  } else if (async && tryTHPCopy(v, dst, src, false, broadcast)) {
    return true;
  }
  THPUtils_setError("copy from %s to %s isn't implemented",
      THPUtils_typename(src), THPUtils_typename(dst));
  return false;
}

inline PyObject * THPStorageCopyMethod(const THPCopyList& v, PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *src;
  PyObject *async = Py_False;
  static char *kwlist[] = {"source", "async", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O:copy_", kwlist, &src, &async)) {
    return NULL;
  }
  if (!PyBool_Check(async)) {
    return PyErr_Format(PyExc_TypeError, "copy_() expected bool for argument async (got '%s')",
        Py_TYPE(async)->tp_name);
  }

  if (!THPCopy(v, self, src, async == Py_True, false)) {
    return NULL;
  }

  Py_INCREF(self);
  return self;
}

inline PyObject * THPTensorCopyMethod(const THPCopyList& v, PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *src;
  PyObject *async = Py_False;
  PyObject *broadcast = Py_True;
  static char *kwlist[] = {"source", "async", "broadcast", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO:copy_", kwlist, &src, &async, &broadcast)) {
    return NULL;
  }
  if (!PyBool_Check(async)) {
    return PyErr_Format(PyExc_TypeError, "copy_() expected bool for argument async (got '%s')",
        Py_TYPE(async)->tp_name);
  }
  if (!PyBool_Check(broadcast)) {
    return PyErr_Format(PyExc_TypeError, "copy_() expected bool for argument broadcast (got '%s')",
        Py_TYPE(broadcast)->tp_name);
  }

  if (!THPCopy(v, self, src, async == Py_True, broadcast == Py_True)) {
    return NULL;
  }

  Py_INCREF(self);
  return self;
}

template <typename StorageDst, typename StorageSrc>
void THPInsertStorageCopyFunction(
  THPCopyList& copyList,
  void (*copyFunc)(LIBRARY_STATE_TYPE StorageDst* x, StorageSrc* z),
  bool async=false)
{
  auto wrapper = [copyFunc](PyObject* dst_, PyObject* src_, bool broadcast) {
    StorageDst* dst = THPTypeInfo<StorageDst>::cdata(dst_);
    StorageSrc* src = THPTypeInfo<StorageSrc>::cdata(src_);

    PyThreadState *_save = NULL;
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

  PyTypeObject* srcType = THPTypeInfo<StorageSrc>::pyType();
  copyList.push_back({ srcType, wrapper, async, false });
}

template <typename TensorDst, typename TensorSrc>
void THPInsertTensorCopyFunction(
  THPCopyList& copyList,
  void (*copyFunc)(LIBRARY_STATE_TYPE TensorDst* x, TensorSrc* z),
  bool async=false,
  bool broadcast=true)
{
  auto wrapper = [copyFunc](PyObject* dst_, PyObject* src_, bool broadcast) {
    TensorDst* dst = THPTypeInfo<TensorDst>::cdata(dst_);
    TensorSrc* src = THPTypeInfo<TensorSrc>::cdata(src_);

    TensorSrc *src_save = src;
    THPPointer<TensorSrc> src_guard(newForExpand<TensorSrc>(LIBRARY_STATE_NOARGS));

    if (broadcast) {
      int ret = expand_inplace1<TensorSrc, TensorDst>(LIBRARY_STATE src_guard.get(), src, dst, "src", "dst", true);
      if (ret == 0) {
        src = src_guard.get();
      }
    }

    PyThreadState *_save = NULL;
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
    src = src_save;
  };

  PyTypeObject* srcType = THPTypeInfo<TensorSrc>::pyType();
  copyList.push_back({ srcType, wrapper, async, broadcast });
}

#endif

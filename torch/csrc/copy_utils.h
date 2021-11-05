#pragma once

#include <functional>
#include <vector>
#include <torch/csrc/Types.h>

typedef std::function<void(PyObject*, PyObject*, bool)> THPCopyFunction;
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
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

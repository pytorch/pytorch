#pragma once

#include <Python.h>
#include <sstream>
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_numbers.h"

inline void THPUtils_packInt64Array(PyObject *tuple, size_t size, const int64_t *sizes) {
  for (size_t i = 0; i != size; ++i) {
    PyObject *i64 = THPUtils_packInt64(sizes[i]);
    if (!i64) {
      std::ostringstream oss;
      oss << "Could not pack int64 at position " << i;
      throw std::runtime_error(oss.str());
    }
    PyTuple_SET_ITEM(tuple, i, THPUtils_packInt64(sizes[i]));
  }
}

inline PyObject* THPUtils_packInt64Array(size_t size, const int64_t *sizes) {
  THPObjectPtr tuple(PyTuple_New(size));
  THPUtils_packInt64Array(tuple.get(), size, sizes);
  return tuple.release();
}

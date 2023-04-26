#pragma once

#include <Python.h>

template <PyCFunctionWithKeywords func>
inline PyCFunction castPyCFunctionWithKeywords() {
  return +[](PyObject* self, PyObject* args) {
    return func(self, args, /*kwargs=*/nullptr); 
  };
}

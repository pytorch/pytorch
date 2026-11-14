#pragma once

#include <c10/macros/Macros.h>
#include <torch/csrc/utils/python_compat.h>

#include <Python.h>

inline PyCFunction castPyCFunctionWithKeywords(PyCFunctionWithKeywords func) {
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wcast-function-type")
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wcast-function-type-strict")
  return reinterpret_cast<PyCFunction>(func);
  C10_DIAGNOSTIC_POP()
  C10_DIAGNOSTIC_POP()
}

#if !IS_PYTHON_3_13_PLUS
using PyCFunctionFast = _PyCFunctionFast;
#endif

inline PyCFunction castPyCFunctionFast(PyCFunctionFast func) {
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wcast-function-type")
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wcast-function-type-strict")
  return reinterpret_cast<PyCFunction>(func);
  C10_DIAGNOSTIC_POP()
  C10_DIAGNOSTIC_POP()
}

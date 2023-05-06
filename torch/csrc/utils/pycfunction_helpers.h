#pragma once

#include <torch/csrc/utils/unsafe_cast_function.h>

#include <Python.h>

inline PyCFunction castPyCFunctionWithKeywords(PyCFunctionWithKeywords func) {
  return torch::unsafe_cast_function<PyCFunction>(func);
}

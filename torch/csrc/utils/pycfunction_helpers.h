#pragma once

#include <Python.h>

inline PyCFunction castPyCFunctionWithKeywords(PyCFunctionWithKeywords func) {
  return (PyCFunction)(void(*)(void))func;
}

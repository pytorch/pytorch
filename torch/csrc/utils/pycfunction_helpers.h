#pragma once

#include <Python.h>

inline PyCFunction castPyCFunctionWithKeywords(PyCFunctionWithKeywords func) {
  // NOLINTNEXTLINE(modernize-redundant-void-arg)
  return (PyCFunction)(void (*)(void))func;
}

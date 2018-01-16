#pragma once

// RAII structs to acquire and release Python's global interpreter lock (GIL)

#include <Python.h>

// Acquires the GIL on construction
struct AutoGIL {
  AutoGIL() : gstate(PyGILState_Ensure()) {
  }
  ~AutoGIL() {
    PyGILState_Release(gstate);
  }

  PyGILState_STATE gstate;
};

// Releases the GIL on construction
struct AutoNoGIL {
  AutoNoGIL() : save(PyEval_SaveThread()) {
  }
  ~AutoNoGIL() {
    PyEval_RestoreThread(save);
  }

  PyThreadState* save;
};

// Runs the function without the GIL
template<typename F>
inline void with_no_gil(F f) {
  AutoNoGIL no_gil;
  f();
}

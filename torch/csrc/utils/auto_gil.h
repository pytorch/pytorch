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

#pragma once

// RAII struct to change the current PyThreadState to a new PyThreadState

#include <torch/csrc/python_headers.h>

// Check this thread local before trying to acqure the GIL using
// PyGILStateEnsure recursively. If the ThreadState has been swapped and the GIL
// hasn't been released, do not try to acquire it again from the same thread
extern thread_local bool swappedThreadState;

struct SwapPyThreadState {
  SwapPyThreadState();

  ~SwapPyThreadState();

  PyThreadState* old_tstate;
  PyThreadState* new_tstate;
};

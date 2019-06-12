#pragma once

// RAII struct to change the current PyThreadState to a new PyThreadState

#include <torch/csrc/python_headers.h>

// Check this thread local before trying to acqure the GIL using
// PyGILStateEnsure recursively. If the ThreadState has been swapped and the GIL
// hasn't been released, do not try to acquire it again from the same thread
static thread_local bool swappedThreadState = false;

struct SwapPyThreadState {
  SwapPyThreadState() : old_tstate(PyThreadState_Get()),
   new_tstate(PyThreadState_New(old_tstate->interp)) {
     PyThreadState_Swap(new_tstate);
     swappedThreadState = true;
   }

  ~SwapPyThreadState() {
      PyThreadState_Swap(old_tstate);
      PyThreadState_Clear(new_tstate);
      PyThreadState_Delete(new_tstate);
  }

  PyThreadState* old_tstate;
  PyThreadState* new_tstate;
};

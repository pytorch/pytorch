#include <torch/csrc/utils/swap_thread_state.h>

thread_local bool swappedThreadState = false;

SwapPyThreadState::SwapPyThreadState() : old_tstate(PyThreadState_Get()),
 new_tstate(PyThreadState_New(old_tstate->interp)) {
   PyThreadState_Swap(new_tstate);
   swappedThreadState = true;
 }

SwapPyThreadState::~SwapPyThreadState() {
  // Copy exception to the old thread state
  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);

  PyThreadState_Swap(old_tstate);
  PyErr_Restore(type, value, traceback);

  PyThreadState_Clear(new_tstate);
  PyThreadState_Delete(new_tstate);
  swappedThreadState = false;

}

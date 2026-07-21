# PyTorch TorchDynamo Frame Hook API

## Overview
This documentation describes the frame hook system used by Dynamo as an
alternative to PEP 523 and the eval frame API.

## Purpose and motivation

Rather than creating and evaluating a frame ourselves, we let CPython do the
heavy lifting of frame management and execution. The hook API contract is to
return a code object to be executed instead of the original code object. This
allows us to not worry about frame lifecycle management, and focus on code
transformation and optimization.

Also, most of the functions for frame creation are private and not exported. Up
until today, Dynamo has been copying CPython's frame creation logic, which is
brittle and hard to maintain. The hook API allows us to avoid this problem.

## Key Components

### Hook Functions
Similar to the `_PyInterpreter_SetEvalFrameFunc()`, the frame hook API provides
a set of functions to manage frame hooks:
- `_PyInterpreterState_SetFrameHook()`: registers a hook function that is called
  for each frame execution.
- `_PyInterpreterState_ClearFrameHook()`: clears the registered hook function.
- `_PyInterpreterState_HasFrameHook()`: checks if a frame hook is currently
  registered.
- `_PyInterpreterState_GetFrameHook()`: retrieves the currently registered frame
  hook function.

## Changes to Dynamo

### New Functions
- `dynamo_eval_hook_default()`: Default hook function that just returns the
  original code object.
- `dynamo_eval_hook_custom()`: Py_INCREF and returns the new code object.
- `enable_frame_hook_shim()`: Calls `PyInterpreterState_SetFrameHook(..., hook_function)`
- `clear_frame_hook_shim()`: Calls `PyInterpreterState_ClearFrameHook()`
- `hook_function()`: The entry point for the frame hook mechanism.
- The `eval_custom` and `eval_default` lambdas were updated to use the new hook
  functions when `use_frame_hook()` returns true.


### Modified Functions
- `use_frame_hook()`: Check if hook-based evaluation is enabled at runtime.
  To enable, set the environment variable `PYTORCH_USE_FRAME_HOOK=1`.
- `eval_frame_callback_get()` and `eval_frame_callback_set()`: Update the key
  for the get/set functions depending on whether hook-based or PEP 523
  evaluation is used.
- all usages of `clear_old_frame_if_python_312_plus()` are guarded to only be
  called when using PEP 523 APIs.


## WARNING

These changes are experimental and subject to change. They do not reflect the
final design of the frame hook API, which may evolve to support multiple hooks.
Currently, only a single frame hook is supported.

## Compatibility Notes

Code was implemented using the current head of the CPython 3.14 branch as
reference. Future versions of CPython may introduce breaking changes that Dynamo
may not yet support.

## Changes to `Include/internal/pycore_ceval.h`:

The `_PyEval_EvalFrame` function was modified to check for a frame hook before
evaluating the frame:

```diff
static inline PyObject*
_PyEval_EvalFrame(PyThreadState *tstate, _PyInterpreterFrame *frame, int throwflag)
{
    EVAL_CALL_STAT_INC(EVAL_CALL_TOTAL);
+
+   if (_PyInterpreterState_HasFrameHook(tstate->interp)) {
+       return _PyEval_FrameHook(tstate, frame, throwflag);
+   }
+
    if (tstate->interp->eval_frame == NULL) {
        return _PyEval_EvalFrameDefault(tstate, frame, throwflag);
    }
    return tstate->interp->eval_frame(tstate, frame, throwflag);
}
```

## Changes to `Python/ceval.c`:

A new function `_PyEval_FrameHook()` was added to handle frame execution when a
frame hook is registered. This new function is responsible for calling the hook,
creating a shadow frame if the code object has changed, and executing the new
frame.

```c
PyObject *_PyEval_FrameHook(PyThreadState *tstate, _PyInterpreterFrame *frame, int throwflag)
{
  assert(_PyInterpreterState_HasFrameHook(tstate->interp));

  _PyFrameEvalFunction eval_function =
      (tstate->interp->eval_frame ? tstate->interp->eval_frame
                                  : _PyEval_EvalFrameDefault);

  PyCodeObject* new_code = _PyEval_CallHook(tstate, frame);

  if (new_code == NULL) {
      assert(_PyErr_Occurred(tstate));
      return NULL;
  }

  PyCodeObject* old_code = (PyCodeObject*) PyUnstable_InterpreterFrame_GetCode(frame);
  if (new_code == old_code) {
      return eval_function(tstate, frame, throwflag);
  }

  _PyInterpreterFrame *shadow = _Hook_CloneFrame(tstate, frame, new_code);
  if (shadow == NULL) {
      return NULL;
  }

  PyObject *r = eval_function(tstate, shadow, throwflag);
  _PyEval_FrameClearAndPop(tstate, frame);

  return r;
}
```

Additionally, two new helper functions were added to support frame hook
operations:

- `_PyEval_CallHook()`: Invokes the registered frame hook function and returns the transformed code object.
- `_Hook_CloneFrame()`: Creates a shadow frame with the new code object while preserving the original frame's state.

## Changes to _PyFrame:

No changes were made to _PyFrame.

## Changes to _PyInterpreterState:

Similar to the eval frame API, the entry point for the frame hook is also per-interpreter:

```c
typedef PyCodeObject* (*_PyFrameHookFunction)(struct _PyInterpreterFrame *);

typedef struct {
    ...
    _PyFrameHookFunction frame_hook;
} PyInterpreterState;
```


The `_PyInterpreterState` also gets the following functions to manage frame hooks:
* `_PyInterpreterState_AddFrameHook(PyInterpreterState*, _PyFrameHookFunction)`: Sets a hook function for frame execution.
* `_PyInterpreterState_ClearFrameHooks(PyInterpreterState*)`: Clears the currently set frame hook function.
* `_PyInterpreterState_ContainsFrameHook(PyInterpreterState*)`: Checks if a frame hook function is set.
* `_PyInterpreterState_EnableFrameHook(PyInterpreterState*, _PyFrameHookFunction)`: Enable the frame hook globally.
* `_PyInterpreterState_DisableFrameHook(PyInterpreterState*, _PyFrameHookFunction)`: Disable frame hook globally.
* `_PyInterpreterState_IsFrameHookEnabled(PyInterpreterState*, _PyFrameHookFunction)`: Check whether the given frame hook function is enabled or not.

When a frame is added using `_PyInterpreterState_AddFrameHook`, CPython will add
it to a list of frame hooks to be called during frame evaluation.

## Execution Flow

```
Frame Evaluation
    ↓
Has Frame Hook?
    ├─ YES → _PyEval_FrameHook()
    │         ├─ Call hook → get new_code
    │         ├─ new_code == old_code?
    │         │  ├─ YES → Execute directly
    │         │  └─ NO → Clone frame with new code
    │         ├─ Execute shadow frame
    │         └─ Clean up original frame
    │
    └─ NO → Check eval_frame override
            ├─ YES → Use custom eval_frame
            └─ NO → Use default evaluator
```

## Implementation

The source code for the frame hook API can be found in the following links:
* CPython: https://github.com/python/cpython/compare/3.14...guilhermeleobas:cpython:guilhermeleobas/hook
* PyTorch: https://github.com/pytorch/pytorch/compare/main...guilhermeleobas:pytorch:guilhermeleobas/proposal

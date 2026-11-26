// CPython 3.12 internals used by the CFG executor (torch/_rust/src/exec). The
// eval-frame hook runs a frame instead of _PyEval_EvalFrameDefault, so it has
// to do the default loop's frame bookkeeping itself.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <Python.h>

#include <pthread.h>
#include <stdint.h>

// see https://github.com/python/cpython/issues/105268
#undef _PyGC_FINALIZED
#define Py_BUILD_CORE
#define Py_BUILD_CORE_MODULE
#include <internal/pycore_atomic.h>
#include <internal/pycore_ceval.h>
#include <internal/pycore_frame.h>
#include <internal/pycore_interp.h>
#include <internal/pycore_pystate.h>
#undef Py_BUILD_CORE
#undef Py_BUILD_CORE_MODULE

#if PY_VERSION_HEX < 0x030C0000 || PY_VERSION_HEX >= 0x030D0000
#error "frame_shim.c supports CPython 3.12 only"
#endif

// ceval.c
#define PY_EVAL_C_STACK_UNITS 2

// Exported from libtorch_python (torch/csrc/dynamo/cpython_defs.c): copies of
// CPython's static _PyFrame_ClearExceptCode / _PyThreadState_PopFrame.
void THP_PyFrame_Clear(_PyInterpreterFrame* frame);
void THP_PyThreadState_PopFrame(PyThreadState* tstate, _PyInterpreterFrame* frame);
PyFrameObject* THP_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame* frame);

PyCodeObject* tr_frame_code(_PyInterpreterFrame* f) {
  return f->f_code;
}

PyObject* tr_frame_qualname(_PyInterpreterFrame* f) {
  return f->f_code->co_qualname;
}

PyObject** tr_frame_localsplus(_PyInterpreterFrame* f) {
  return f->localsplus;
}

PyObject* tr_frame_globals(_PyInterpreterFrame* f) {
  return f->f_globals;
}

PyObject* tr_frame_builtins(_PyInterpreterFrame* f) {
  return f->f_builtins;
}

// May be NULL (function frames have no locals dict).
PyObject* tr_frame_locals(_PyInterpreterFrame* f) {
  return f->f_locals;
}

// Whether `func` is one of the hooks the interpreter calls for the events it
// reports (those registered through `sys`): Python code that runs from
// wherever such an event happens.
int tr_event_hook(PyObject* func) {
  PyObject* hooks = _PyInterpreterState_GET()->audit_hooks;
  if (hooks == NULL || !PyList_CheckExact(hooks)) {
    return 0;
  }
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(hooks); i++) {
    if (PyList_GET_ITEM(hooks, i) == func) {
      return 1;
    }
  }
  return 0;
}

PyObject* tr_frame_func(_PyInterpreterFrame* f) {
  return f->f_funcobj;
}

int tr_frame_owned_by_generator(_PyInterpreterFrame* f) {
  return f->owner == FRAME_OWNED_BY_GENERATOR;
}

// The executor points the frame at code unit `unit`, as if that instruction
// is executing, by storing code_units + unit to prev_instr: tracebacks,
// f_lineno and f_lasti read it.
_Py_CODEUNIT** tr_frame_prev_instr_slot(_PyInterpreterFrame* f) {
  return &f->prev_instr;
}

_Py_CODEUNIT* tr_frame_code_units(_PyInterpreterFrame* f) {
  return _PyCode_CODE(f->f_code);
}

int tr_frame_get_instr(_PyInterpreterFrame* f) {
  return (int)(f->prev_instr - _PyCode_CODE(f->f_code));
}

int tr_frame_nlocalsplus(_PyInterpreterFrame* f) {
  return f->f_code->co_nlocalsplus;
}

// The executor recurses natively for every Python call (and every nested
// generator resumption, since CPython inlines those only without an eval-frame
// hook), so the recursion counters alone don't bound its C stack. Like
// CPython 3.14, refuse to go deeper when little native stack is left.
#define STACK_MARGIN (256 * 1024)
static __thread uintptr_t stack_floor;

static int native_stack_low(void) {
  char here;
  if (stack_floor == 0) {
    pthread_attr_t attr;
    void* base;
    size_t size;
    if (pthread_getattr_np(pthread_self(), &attr) != 0) {
      return 0;
    }
    pthread_attr_getstack(&attr, &base, &size);
    pthread_attr_destroy(&attr);
    stack_floor = (uintptr_t)base + STACK_MARGIN;
  }
  return (uintptr_t)&here < stack_floor;
}

// Makes the frame the thread's current frame (what sys._getframe() and
// tracebacks see) and charges the recursion limits, like the start of
// _PyEval_EvalFrameDefault. Every executor frame is a C-level eval entry (the
// executor recurses natively), hence the C-stack charge. Returns -1 with
// RecursionError set if a limit is hit; either way, leave with tr_frame_exit,
// tr_gen_exit or tr_gen_suspend.
int tr_frame_enter(PyThreadState* tstate, _PyInterpreterFrame* f) {
  f->previous = tstate->cframe->current_frame;
  tstate->cframe->current_frame = f;
  tstate->c_recursion_remaining -= (PY_EVAL_C_STACK_UNITS - 1);
  if (tstate->c_recursion_remaining-- <= 0 && _Py_CheckRecursiveCall(tstate, "")) {
    tstate->c_recursion_remaining--;
    tstate->py_recursion_remaining--;
    return -1;
  }
  // _Py_EnterRecursivePy / _Py_CheckRecursiveCallPy, which aren't exported.
  tstate->py_recursion_remaining--;
  if (native_stack_low()) {
    PyErr_SetString(PyExc_RecursionError, "maximum recursion depth exceeded");
    return -1;
  }
  if (tstate->py_recursion_remaining + 1 <= 0) {
    if (tstate->recursion_headroom) {
      if (tstate->py_recursion_remaining < -50) {
        Py_FatalError("Cannot recover from Python stack overflow.");
      }
    } else {
      tstate->recursion_headroom++;
      PyErr_SetString(PyExc_RecursionError, "maximum recursion depth exceeded");
      tstate->recursion_headroom--;
      return -1;
    }
  }
  return 0;
}

// Returns the recursion charges taken by tr_frame_enter and unlinks the frame.
static void frame_leave(PyThreadState* tstate, _PyInterpreterFrame* f) {
  tstate->cframe->current_frame = f->previous;
  tstate->py_recursion_remaining++;
  tstate->c_recursion_remaining += PY_EVAL_C_STACK_UNITS;
}

// For a thread-owned frame that was never entered.
void tr_frame_clear_and_pop(PyThreadState* tstate, _PyInterpreterFrame* f) {
  THP_PyFrame_Clear(f);
  THP_PyThreadState_PopFrame(tstate, f);
}

// Undoes tr_frame_enter, then clears and pops a thread-owned frame (the
// default loop's _PyEvalFrameClearAndPop).
void tr_frame_exit(PyThreadState* tstate, _PyInterpreterFrame* f) {
  frame_leave(tstate, f);
  tr_frame_clear_and_pop(tstate, f);
}

// Adds the frame to the pending exception's traceback (the default loop's
// `error:` label). Returns -1 if that fails.
// The innermost complete frame of the thread, or NULL.
_PyInterpreterFrame* tr_current_frame(PyThreadState* tstate) {
  _PyInterpreterFrame* f = tstate->cframe->current_frame;
  while (f != NULL && _PyFrame_IsIncomplete(f)) {
    f = f->previous;
  }
  return f;
}

int tr_traceback_here(_PyInterpreterFrame* f) {
  if (_PyFrame_IsIncomplete(f)) {
    return 0;
  }
  PyFrameObject* frame_obj = f->frame_obj;
  if (frame_obj == NULL) {
    frame_obj = THP_PyFrame_MakeAndSetFrameObject(f);
    if (frame_obj == NULL) {
      return -1;
    }
  }
  return PyTraceBack_Here(frame_obj);
}

// sys.settrace / sys.setprofile / sys.monitoring: the executor emits no
// events, so it refuses to run while any are active.
int tr_tracing_active(PyThreadState* tstate) {
  if (tstate->c_tracefunc != NULL || tstate->c_profilefunc != NULL) {
    return 1;
  }
  _Py_GlobalMonitors* m = &tstate->interp->monitors;
  for (int i = 0; i < _PY_MONITORING_UNGROUPED_EVENTS; i++) {
    if (m->tools[i]) {
      return 1;
    }
  }
  return 0;
}

// --- Generator frames ---
//
// A suspended frame keeps its value stack in localsplus[nlocalsplus:stacktop],
// exactly like CPython's YIELD_VALUE, so either engine can resume it.

static PyGenObject* frame_gen(_PyInterpreterFrame* f) {
  return _PyFrame_GetGenerator(f);
}

// Pushes `obj` (stolen) onto the frame's value stack.
void tr_frame_push(_PyInterpreterFrame* f, PyObject* obj) {
  f->localsplus[f->stacktop++] = obj;
}

// Pops and returns (as a new reference) the top of the frame's value stack.
PyObject* tr_frame_pop(_PyInterpreterFrame* f) {
  return f->localsplus[--f->stacktop];
}

int tr_frame_stack_depth(_PyInterpreterFrame* f) {
  return f->stacktop - f->f_code->co_nlocalsplus;
}

// YIELD_VALUE's bookkeeping, after the stack has been saved: the generator is
// suspended at instruction `unit` and the frame is unlinked.
void tr_gen_suspend(PyThreadState* tstate, _PyInterpreterFrame* f, int unit) {
  PyGenObject* gen = frame_gen(f);
  gen->gi_frame_state = FRAME_SUSPENDED;
  f->prev_instr = _PyCode_CODE(f->f_code) + unit;
  tstate->exc_info = gen->gi_exc_state.previous_item;
  gen->gi_exc_state.previous_item = NULL;
  frame_leave(tstate, f);
  f->previous = NULL;
}

// Undoes tr_frame_enter for a generator frame that returned or raised
// (clear_gen_frame in ceval.c).
void tr_gen_exit(PyThreadState* tstate, _PyInterpreterFrame* f) {
  frame_leave(tstate, f);
  PyGenObject* gen = frame_gen(f);
  gen->gi_frame_state = FRAME_CLEARED;
  tstate->exc_info = gen->gi_exc_state.previous_item;
  gen->gi_exc_state.previous_item = NULL;
  // THP_PyFrame_Clear consumes a code reference (dropping it, or handing it to
  // a frame object that outlives the frame). A generator frame's reference
  // belongs to the generator, so give it one of its own. (CPython's
  // _PyFrame_ClearExceptCode isn't exported.)
  Py_INCREF(f->f_code);
  THP_PyFrame_Clear(f);
  f->previous = NULL;
}

// co_localsplusnames: the slot names in frame order. Unlike co_varnames (and
// co_cellvars/co_freevars), reading it doesn't cache new tuples on the code
// object, which would be visible to gc.get_objects().
PyObject* tr_code_localsplusnames(PyCodeObject* code) {
  return code->co_localsplusnames;
}

// Whether CPython's eval breaker is set: a signal, pending call, scheduled
// GC, GIL drop request or async exception is waiting.
// The interpreter's eval-breaker flag (an _Py_atomic_int), read relaxed.
int* tr_eval_breaker_flag(PyThreadState* tstate) {
  return (int*)&tstate->interp->ceval.eval_breaker;
}

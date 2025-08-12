#define PY_SSIZE_T_CLEAN
#include <opcode.h>
#include <signal.h>
#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/cpp_shim.h>
#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/dynamo/eval_frame_cpp.h>
#include <torch/csrc/utils/python_compat.h>

PyObject* guard_error_hook = NULL;
PyObject* guard_complete_hook = NULL;

typedef struct {
  int active_dynamo_threads;
} ModuleState;

// static int active_dynamo_threads = 0;

static Py_tss_t eval_frame_callback_key = Py_tss_NEEDS_INIT;

static PyObject* eval_frame_callback_get(void) {
  void* result = PyThread_tss_get(&eval_frame_callback_key);
  if (unlikely(result == NULL)) {
    return (PyObject*)Py_None;
  } else {
    return (PyObject*)result;
  }
}

void eval_frame_callback_set(PyObject* obj) {
  PyThread_tss_set(&eval_frame_callback_key, obj);
}

// 3.14 Not supported at all. See cpython_defs.c for hints
#if !(IS_PYTHON_3_14_PLUS)

#define DECLARE_PYOBJ_ATTR(name)                        \
  static PyObject* THPPyInterpreterFrame_##name(        \
      THPPyInterpreterFrame* self, PyObject* _noargs) { \
    PyObject* res = (PyObject*)self->frame->name;       \
    Py_XINCREF(res);                                    \
    return res;                                         \
  }

DECLARE_PYOBJ_ATTR(f_globals)
DECLARE_PYOBJ_ATTR(f_builtins)

static PyObject* THPPyInterpreterFrame_f_locals(
    THPPyInterpreterFrame* self,
    PyObject* _noargs) {
  DEBUG_NULL_CHECK(self->locals);
  Py_XINCREF(self->locals);
  return self->locals;
}

#if IS_PYTHON_3_13_PLUS
DECLARE_PYOBJ_ATTR(f_executable)
#else
DECLARE_PYOBJ_ATTR(f_code)
#endif

#undef DECLARE_PYOBJ_ATTR

// This is not a true attribute of the class but we do access it in python and
// it is hard to implement on the python side, so do it here:
static PyObject* THPPyInterpreterFrame_f_lasti(
    THPPyInterpreterFrame* self,
    PyObject* _noargs) {
#if IS_PYTHON_3_11_PLUS
  return PyLong_FromLong(_PyInterpreterFrame_LASTI(self->frame));
#else
  return PyLong_FromLong(self->frame->f_lasti);
#endif // IS_PYTHON_3_11_PLUS
}

static PyObject* THPPyInterpreterFrame_f_lineno(
    THPPyInterpreterFrame* self,
    PyObject* _noargs) {
#if IS_PYTHON_3_11_PLUS
  if (!self->frame->frame_obj) {
    return PyLong_FromLong(F_CODE(self->frame)->co_firstlineno);
  }
  int lineno = PyFrame_GetLineNumber(self->frame->frame_obj);
  if (lineno < 0) {
    Py_RETURN_NONE;
  }
  return PyLong_FromLong(lineno);
#else
  return PyLong_FromLong(self->frame->f_lineno);
#endif // IS_PYTHON_3_11_PLUS
}

static PyObject* THPPyInterpreterFrame_f_back(
    THPPyInterpreterFrame* self,
    PyObject* _noargs) {
#if IS_PYTHON_3_11_PLUS
  if (!self->frame->frame_obj) {
    Py_RETURN_NONE;
  }
  return (PyObject*)PyFrame_GetBack(self->frame->frame_obj);
#else
  return Py_XNewRef(self->frame->f_back);
#endif // IS_PYTHON_3_11_PLUS
}

static PyObject* THPPyInterpreterFrame_closure(
    THPPyInterpreterFrame* self,
    PyObject* _noargs) {
#if IS_PYTHON_3_12_PLUS
  PyObject* closure = ((PyFunctionObject*)self->frame->f_funcobj)->func_closure;
  return closure == NULL ? PyTuple_New(0) : Py_XNewRef(closure);
#elif IS_PYTHON_3_11_PLUS
  PyObject* closure = ((PyFunctionObject*)self->frame->f_func)->func_closure;
  return closure == NULL ? PyTuple_New(0) : Py_XNewRef(closure);
#else
  PyCodeObject* code = self->frame->f_code;
  // Why this check? See
  // https://github.com/python/cpython/blob/5f24da9d75bb0150781b17ee4706e93e6bb364ea/Objects/frameobject.c#L1058-L1065
  if (code->co_flags & CO_OPTIMIZED) {
    int size = PyTuple_GET_SIZE(code->co_freevars);
    PyObject* freevars = PyTuple_New(size);
    int ncells = PyTuple_GET_SIZE(code->co_cellvars);
    PyObject** freevarArr =
        self->frame->f_localsplus + code->co_nlocals + ncells;
    for (int i = 0; i < size; i++) {
      PyTuple_SET_ITEM(freevars, i, Py_XNewRef(freevarArr[i]));
    }
    return freevars;
  }
  return PyTuple_New(0);
#endif // IS_PYTHON_3_11_PLUS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static struct PyGetSetDef THPPyInterpreterFrame_properties[] = {
    {"f_globals", (getter)THPPyInterpreterFrame_f_globals, NULL, NULL, NULL},
    {"f_builtins", (getter)THPPyInterpreterFrame_f_builtins, NULL, NULL, NULL},
    {"f_locals", (getter)THPPyInterpreterFrame_f_locals, NULL, NULL, NULL},
#if IS_PYTHON_3_13_PLUS
    {"f_code", (getter)THPPyInterpreterFrame_f_executable, NULL, NULL, NULL},
#else
    {"f_code", (getter)THPPyInterpreterFrame_f_code, NULL, NULL, NULL},
#endif
    {"f_lasti", (getter)THPPyInterpreterFrame_f_lasti, NULL, NULL, NULL},
    {"f_lineno", (getter)THPPyInterpreterFrame_f_lineno, NULL, NULL, NULL},
    {"f_back", (getter)THPPyInterpreterFrame_f_back, NULL, NULL, NULL},
    {"closure", (getter)THPPyInterpreterFrame_closure, NULL, NULL, NULL},
    {NULL}};

static PyTypeObject THPPyInterpreterFrameType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "torch._C._dynamo.eval_frame._PyInterpreterFrame",
    .tp_basicsize = sizeof(THPPyInterpreterFrame),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = THPPyInterpreterFrame_properties,
};

THPPyInterpreterFrame* THPPyInterpreterFrame_New(
    THP_EVAL_API_FRAME_OBJECT* frame) {
  PyTypeObject* type = (PyTypeObject*)&THPPyInterpreterFrameType;
  THPPyInterpreterFrame* self = (THPPyInterpreterFrame*)type->tp_alloc(type, 0);
  if (!self)
    return NULL;
  self->frame = frame;
  self->locals = NULL;
  return self;
}

static PyObject* dynamo__custom_eval_frame_shim(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag);
static PyObject* (*previous_eval_frame)(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) = NULL;

static PyObject* dynamo_custom_eval_frame_shim(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) {
  return dynamo__custom_eval_frame_shim(tstate, frame, throw_flag);
}

PyObject* dynamo_eval_frame_default(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) {
  if (tstate == NULL) {
    tstate = PyThreadState_GET();
  }
  if (previous_eval_frame) {
    return previous_eval_frame(tstate, frame, throw_flag);
  } else {
    return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
  }
}

static void enable_eval_frame_shim(PyThreadState* tstate) {
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
      &dynamo_custom_eval_frame_shim) {
    DEBUG_CHECK(previous_eval_frame == NULL);
    previous_eval_frame = _PyInterpreterState_GetEvalFrameFunc(tstate->interp);
    _PyInterpreterState_SetEvalFrameFunc(
        tstate->interp, &dynamo_custom_eval_frame_shim);
  }
}

static void enable_eval_frame_default(PyThreadState* tstate) {
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
      previous_eval_frame) {
    DEBUG_CHECK(previous_eval_frame != NULL);
    _PyInterpreterState_SetEvalFrameFunc(tstate->interp, previous_eval_frame);
    previous_eval_frame = NULL;
  }
}

const char* get_frame_name(THP_EVAL_API_FRAME_OBJECT* frame) {
  // Returns the C string name of the current frame.
  DEBUG_CHECK(PyUnicode_Check(F_CODE(frame)->co_name));
  return PyUnicode_AsUTF8(F_CODE(frame)->co_name);
}

static PyObject* dynamo_eval_custom_code_impl(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    PyCodeObject* code,
    int throw_flag) {
  DEBUG_NULL_CHECK(tstate);
  DEBUG_NULL_CHECK(frame);
  DEBUG_NULL_CHECK(code);

#if IS_PYTHON_3_11_PLUS

  // Generate Python function object and _PyInterpreterFrame in a way similar to
  // https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Python/ceval.c#L1130
#if IS_PYTHON_3_12_PLUS
  PyFunctionObject* old_func = (PyFunctionObject*)frame->f_funcobj;
  size_t size = code->co_framesize;
#else
  PyFunctionObject* old_func = frame->f_func;
  size_t size = code->co_nlocalsplus + code->co_stacksize + FRAME_SPECIALS_SIZE;
#endif

  PyFunctionObject* func = _PyFunction_CopyWithNewCode(old_func, code);
  if (func == NULL) {
    return NULL;
  }

  THP_EVAL_API_FRAME_OBJECT* shadow =
      THP_PyThreadState_BumpFramePointerSlow(tstate, size);
  if (shadow == NULL) {
    Py_DECREF(func);
    return NULL;
  }

  Py_INCREF(func);
  // consumes reference to func
#if IS_PYTHON_3_12_PLUS
  _PyFrame_Initialize(shadow, func, NULL, code, 0);
#else
  _PyFrame_InitializeSpecials(shadow, func, NULL, code->co_nlocalsplus);
#endif

  PyObject** fastlocals_old = frame->localsplus;
  PyObject** fastlocals_new = shadow->localsplus;
  Py_ssize_t n_old = F_CODE(frame)->co_nlocalsplus;
  Py_ssize_t n_new = code->co_nlocalsplus;

  // localsplus are XINCREF'd by default eval frame, so all values must be
  // valid.
#if !(IS_PYTHON_3_12_PLUS)
  // _PyFrame_Initialize in 3.12 already does this
  for (int i = 0; i < code->co_nlocalsplus; i++) {
    fastlocals_new[i] = NULL;
  }
#endif

#else

  THP_EVAL_API_FRAME_OBJECT* shadow =
      PyFrame_New(tstate, code, frame->f_globals, NULL);
  if (shadow == NULL) {
    return NULL;
  }

  PyObject** fastlocals_old = frame->f_localsplus;
  PyObject** fastlocals_new = shadow->f_localsplus;
  Py_ssize_t n_old = F_CODE(frame)->co_nlocals +
      PyCode_GetNFreevars(F_CODE(frame)) + PyCode_GetNCellvars(F_CODE(frame));
  Py_ssize_t n_new =
      code->co_nlocals + PyCode_GetNFreevars(code) + PyCode_GetNCellvars(code);

#endif

  // ============== Initialize new frame from old frame ============
  // Python internal for executing a function:
  //  1. CPython interpreter first creates an empty frame according to the code
  //  object
  //  2. CPython interpreter initializes the frame by filling arguments/free
  //  variables into frame and initializing cell variables
  //  3. CPython interpreter executes the code object
  //
  // Dynamo hooks the 3th step: before executing the code object, Dynamo
  // transforms the code object into a new code object. Then, the old frame is
  // not suitable for executing the new code. Therefore, Dynamo needs to
  // manually create and initialize a new frame to execute the new code. The
  // main task is to copy data in old frame to new frame, concerning a storage
  // space named `localsplus`.
  //
  // localsplus storage is an array with the following layout:
  // |   args   |   new_locals    |    cell_variables |   free_variables    |
  // | <--- from left to right, index from 0 to n - 1 ---> |
  // code.co_varnames == args + new_locals, code.co_nlocals ==
  // len(code.co_varnames) code.co_freevars == free_variables In Python 3.10 and
  // lower, `n == code.co_nlocals + len(code.co_cellvars) +
  // len(code.co_freevars)` (Python expression) In Python 3.11 and higher, `n <=
  // code.co_nlocals + len(code.co_cellvars) + len(code.co_freevars)` (Python
  // expression). There is an extra field in Python C-API: `n ==
  // code->co_nlocalsplus` (C expression) to retrieve the length of array. The
  // complexity happens if an argument becomes a cell variable:
  //  In Python 3.10 and lower, `code.co_cellvars == cell_variables`, and the
  //  corresponding slot in args becomes `NULL`. In Python 3.11 and higher,
  //  `code.co_cellvars > cell_variables`, that cell variable is still stored in
  //  args, with a flag set in corresponding item's `co_localspluskinds` .
  //
  // ideally, we need to look up new localsplus from old localsplus by name:
  // for i, name, value in enumerate(localsplusnames_old):
  //   if value != NULL: (NULL happens for new local variables and arguments
  //   that becomes cell variables)
  //     name_to_idx[name] = i
  // for i, name in enumerate(localsplusnames_new):
  //  if name in name_to_idx:
  //    fastlocals_new[i] = fastlocals_old[name_to_idx[name]]
  //
  // The above process of building a `name_to_idx` mapping is expensive.
  // Dynamo makes the following assumptions:
  //  1. new code has the same arguments as the old code (both the number and
  //  the order)
  //  2. new code has the same cell variables as the old code (both the number
  //  and the order)
  //  3. new code has the same free variables as the old code (both the number
  //  and the order) The only flexibility lies in new local variables: new code
  //  can introduce their own variables.
  // With these assumptions, Dynamo can copy data directly by index. Dynamo just
  // needs to take care of copying cell variables correctly. To avoid runtime
  // cost, the assumptions are checked when we first generate the code object in
  // pytorch/torch/_dynamo/convert_frame.py .

  // copy args
  // according to https://docs.python.org/3/library/inspect.html , `co_argcount`
  // is the number of arguments (not including keyword only arguments, * or **
  // args). so we need to add `co_kwonlyargcount` and `co_flags` to get the
  // total number of arguments.
  // !!(F_CODE(frame)->co_flags & CO_VARARGS) is 1 if the function has *args, 0
  // otherwise
  // !!(F_CODE(frame)->co_flags & CO_VARKEYWORDS) is 1 if the function has
  // **kwargs, 0 otherwise they convert bit flags to 0 or 1, and avoid
  // branching. This is performance critical code, so we really care about
  // performance.
  Py_ssize_t total_argcount_old = F_CODE(frame)->co_argcount +
      F_CODE(frame)->co_kwonlyargcount +
      !!(F_CODE(frame)->co_flags & CO_VARARGS) +
      !!(F_CODE(frame)->co_flags & CO_VARKEYWORDS);

  for (Py_ssize_t i = 0; i < total_argcount_old; i++) {
    Py_XINCREF(fastlocals_old[i]);
    fastlocals_new[i] = fastlocals_old[i];
  }

  // copy free vars
  Py_ssize_t nfrees_old = PyCode_GetNFreevars(F_CODE(frame));

  for (Py_ssize_t i = 0; i < nfrees_old; i++) {
    Py_XINCREF(fastlocals_old[n_old - 1 - i]);
    fastlocals_new[n_new - 1 - i] = fastlocals_old[n_old - 1 - i];
  }

  // copy cell vars, from high index to low index, until it meets a variable
  // that is not cell variable.
  for (Py_ssize_t i = n_old - nfrees_old - 1, j = n_new - nfrees_old - 1;
       i >= total_argcount_old;
       i--, j--) {
    // conditional test to tell if a variable is not a cell variable
    // this is straightforward in Python 3.11 and higher, as there are bit flags
    // in `co_localspluskinds` to tell if a variable is a cell variable. in
    // Python 3.10 and lower, essentially we are checking if a variable is a new
    // local variable (because of the layout mentioned above, the first variable
    // that is not cell variable is the first new local variable). the
    // corresponding slot in `flocalsplus` is NULL for new local variables.
#if IS_PYTHON_3_11_PLUS
    if (!(_PyLocals_GetKind(F_CODE(frame)->co_localspluskinds, i) &
          CO_FAST_CELL)) {
      break;
    }
#else
    if (fastlocals_old[i] == NULL) {
      break;
    }
#endif

    Py_XINCREF(fastlocals_old[i]);
    fastlocals_new[j] = fastlocals_old[i];
  }

  // NOTE: if you want to evaluate frame instead of shadow in 3.12+,
  // you need to clear_old_frame_if_python_312_plus the shadow frame BEFORE
  // calling eval_frame_default (i.e. here) and comment out the
  // clear_old_frame_if_python_312_plus call on the original frame.

  PyObject* result = dynamo_eval_frame_default(tstate, shadow, throw_flag);

#if IS_PYTHON_3_12_PLUS

  // frame is cleared by caller
  Py_DECREF(func);

#elif IS_PYTHON_3_11_PLUS

  // In 3.11, shadow has is_entry set to true, so _PyEvalFrameClearAndPop is not
  // called, so we manually clear and pop the shadow frame.
  THP_PyFrame_Clear(shadow);
  THP_PyThreadState_PopFrame(tstate, shadow);
  Py_DECREF(func);

#else

  Py_DECREF(shadow);

#endif

  return result;
}

// This wrapper function adds a profiler event
PyObject* dynamo_eval_custom_code(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    PyCodeObject* code,
    const char* trace_annotation,
    int throw_flag) {
  _PytorchRecordFunctionState* rf =
      _pytorch_record_function_enter(trace_annotation);
  PyObject* result =
      dynamo_eval_custom_code_impl(tstate, frame, code, throw_flag);
  _pytorch_record_function_exit(rf);
  return result;
}

static PyObject* dynamo__custom_eval_frame_shim(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) {
  // Shims logic into one of three states. Can probably be refactored into a
  // single func, later:
  //  - None: disables TorchDynamo
  //  - False: run-only mode (reuse existing compiles)
  //  - Python callable(): enables TorchDynamo
  PyObject* callback = eval_frame_callback_get();

  if (callback == Py_None) {
    return dynamo_eval_frame_default(tstate, frame, throw_flag);
  }

  return dynamo__custom_eval_frame(tstate, frame, throw_flag, callback);
}

#else // !(IS_PYTHON_3_14_PLUS)

// Fake definitions for everything we removed

static void enable_eval_frame_shim(PyThreadState* tstate) {}
static void enable_eval_frame_default(PyThreadState* tstate) {}
PyObject* dynamo_eval_custom_code(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    PyCodeObject* code,
    const char* trace_annotation,
    int throw_flag) {}
THPPyInterpreterFrame* THPPyInterpreterFrame_New(
    THP_EVAL_API_FRAME_OBJECT* frame) {}
PyObject* dynamo_eval_frame_default(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) {}

static struct PyGetSetDef THPPyInterpreterFrame_properties[] = {NULL};

static PyTypeObject THPPyInterpreterFrameType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "torch._C._dynamo.eval_frame._PyInterpreterFrame",
    .tp_basicsize = sizeof(THPPyInterpreterFrame),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = THPPyInterpreterFrame_properties,
};

#endif // !(IS_PYTHON_3_14_PLUS)

void clear_old_frame_if_python_312_plus(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame) {
#if IS_PYTHON_3_12_PLUS

  THP_PyFrame_Clear(frame);
  THP_PyThreadState_PopFrame(tstate, frame);

#endif
}

static PyObject* increment_working_threads(
    PyThreadState* tstate,
    PyObject* module) {
  ModuleState* state = PyModule_GetState(module);

  if (state != NULL) {
    state->active_dynamo_threads = state->active_dynamo_threads + 1;
    if (state->active_dynamo_threads > 0) {
      enable_eval_frame_shim(tstate);
    }
  }

  Py_RETURN_NONE;
}

static PyObject* decrement_working_threads(
    PyThreadState* tstate,
    PyObject* module) {
  ModuleState* state = PyModule_GetState(module);

  if (state != NULL) {
    if (state->active_dynamo_threads > 0) {
      state->active_dynamo_threads = state->active_dynamo_threads - 1;
      if (state->active_dynamo_threads == 0) {
        enable_eval_frame_default(tstate);
      }
    }
  }

  Py_RETURN_NONE;
}

static PyObject* set_eval_frame(
    PyObject* new_callback,
    PyThreadState* tstate,
    PyObject* module) {
  // Change the eval frame callback and return the old one
  //  - None: disables TorchDynamo
  //  - False: run-only mode (reuse existing compiles)
  //  - Python callable(): enables TorchDynamo
  PyObject* old_callback = eval_frame_callback_get();

  // owned by caller
  Py_INCREF(old_callback);

  if (old_callback != Py_None && new_callback == Py_None) {
    decrement_working_threads(tstate, module);
  } else if (old_callback == Py_None && new_callback != Py_None) {
    increment_working_threads(tstate, module);
  }

  Py_INCREF(new_callback);
  Py_DECREF(old_callback);

  // Set thread local callback. This will drive behavior of our shim, if/when it
  // is installed.
  eval_frame_callback_set(new_callback);

  return old_callback;
}

static PyObject* set_eval_frame_py(PyObject* module, PyObject* callback) {
  if (callback != Py_None && callback != Py_False &&
      !PyCallable_Check(callback)) {
    DEBUG_TRACE0("arg error");
    PyErr_SetString(PyExc_TypeError, "expected a callable");
    return NULL;
  }
  DEBUG_TRACE(
      "python enabled=%d and is run_only=%d",
      callback != Py_None,
      callback == Py_False);
  return set_eval_frame(callback, PyThreadState_GET(), module);
}

static PyObject* set_skip_guard_eval_unsafe(
    PyObject* dummy,
    PyObject* skip_guard_unsafe_flag) {
  if (skip_guard_unsafe_flag != Py_False && skip_guard_unsafe_flag != Py_True) {
    DEBUG_TRACE0("arg error");
    PyErr_SetString(PyExc_TypeError, "expected True/False");
    return NULL;
  }
  bool old_skip_guard_eval_unsafe = is_skip_guard_eval_unsafe;
  is_skip_guard_eval_unsafe = skip_guard_unsafe_flag == Py_True;
  if (old_skip_guard_eval_unsafe) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject* get_eval_frame_callback_py(PyObject* dummy, PyObject* args) {
  // New reference
  PyObject* callback = eval_frame_callback_get();
  Py_INCREF(callback);
  return callback;
}

static PyObject* reset_code(PyObject* dummy, PyObject* code) {
  if (!PyCode_Check(code)) {
    DEBUG_TRACE0("arg error");
    PyErr_SetString(PyExc_TypeError, "expected a code object");
    return NULL;
  }

  // set_extra_state destroys the existing object on extra scratch space.
  set_extra_state((PyCodeObject*)code, NULL);
  Py_RETURN_NONE;
}

static PyObject* unsupported(PyObject* dummy, PyObject* args) {
  // a dummy C function used in testing
  PyObject* obj1 = NULL;
  PyObject* obj2 = NULL;
  if (!PyArg_ParseTuple(args, "OO", &obj1, &obj2)) {
    return NULL;
  }
  Py_INCREF(obj2);
  return obj2;
}

static PyObject* set_guard_error_hook(PyObject* dummy, PyObject* obj) {
  if (obj == Py_None) {
    obj = NULL;
  }
  Py_XSETREF(guard_error_hook, Py_XNewRef(obj));
  Py_RETURN_NONE;
}

static PyObject* set_guard_complete_hook(PyObject* dummy, PyObject* obj) {
  PyObject* old_hook = guard_complete_hook;

  if (obj == Py_None) {
    obj = NULL;
  }

  guard_complete_hook = Py_XNewRef(obj);

  if (old_hook == NULL) {
    Py_RETURN_NONE;
  } else {
    return old_hook;
  }
}

// Debugging function for GNU C only.
// Used to set gdb breakpoints in hot CPython sites from Python.
// Code example:
//
// def foo(x):
//     x = x + 1
//     torch._dynamo.eval_frame.raise_sigtrap()
//     # (gdb) b bytecodes.c:1234 (whatever line CALL is handled)
//     x = torch.sin(x)  # gdb breakpoint hit when sin is called
//
// In this example, we want to breakpoint on CALL in bytecodes.c only when
// running foo. Otherwise, we would need to breakpoint before running the
// program, and that breakpoint would be hit every time Python makes a function
// call, leading to a spammy debugging experience.
static PyObject* raise_sigtrap(PyObject* dummy, PyObject* obj) {
#ifdef __GNUC__
  raise(SIGTRAP);
#endif
  Py_RETURN_NONE;
}

static int clear_state(PyObject* module) {
  ModuleState* state = PyModule_GetState(module);
  if (state) {
    state->active_dynamo_threads = 0;
    return 0;
  }
  return -1;
}

bool is_skip_guard_eval_unsafe = false;

static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame_py, METH_O, NULL},
    {"set_skip_guard_eval_unsafe", set_skip_guard_eval_unsafe, METH_O, NULL},
    {"get_eval_frame_callback", get_eval_frame_callback_py, METH_NOARGS, NULL},
    {"reset_code", reset_code, METH_O, NULL},
    {"unsupported", unsupported, METH_VARARGS, NULL},
    {"set_code_exec_strategy", set_code_exec_strategy, METH_VARARGS, NULL},
    {"set_guard_error_hook", set_guard_error_hook, METH_O, NULL},
    {"set_guard_complete_hook", set_guard_complete_hook, METH_O, NULL},
    {"raise_sigtrap", raise_sigtrap, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "torch._C._dynamo.eval_frame",
    .m_doc = "Module containing hooks to override eval_frame",
    .m_size = sizeof(ModuleState),
    .m_methods = _methods,
    .m_clear = clear_state};

#if IS_PYTHON_3_12_PLUS
#define _PyEval_RequestCodeExtraIndex PyUnstable_Eval_RequestCodeExtraIndex
#endif

PyObject* torch_c_dynamo_eval_frame_init(void) {
  extra_index = _PyEval_RequestCodeExtraIndex(destroy_extra_state);
  if (extra_index < 0) {
    PyErr_SetString(
        PyExc_RuntimeError, "dynamo: unable to register extra index");
    return NULL;
  }

  int result = PyThread_tss_create(&eval_frame_callback_key);
  CHECK(result == 0);

  Py_INCREF(Py_None);
  eval_frame_callback_set(Py_None);

  PyObject* module = PyModule_Create(&_module);
  if (module == NULL) {
    return NULL;
  }

#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
#endif

  if (PyType_Ready(&THPPyInterpreterFrameType) < 0) {
    return NULL;
  }
  Py_INCREF(&THPPyInterpreterFrameType);
  if (PyModule_AddObject(
          module,
          "_PyInterpreterFrame",
          (PyObject*)&THPPyInterpreterFrameType) != 0) {
    return NULL;
  }

  return module;
}

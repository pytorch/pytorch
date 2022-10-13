#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>

// Only Python 3.7 through 3.10 supported
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 11
#define _PY_VERSION_OK

#include <frameobject.h>
#include <pystate.h>

// see https://bugs.python.org/issue35886
#if PY_VERSION_HEX >= 0x03080000
#define Py_BUILD_CORE
#include <internal/pycore_pystate.h>
#undef Py_BUILD_CORE
#endif

#ifdef _WIN32
#define unlikely(x) (x)
#else
#define unlikely(x) __builtin_expect((x), 0)
#endif

#define NULL_CHECK(val)                                         \
  if (unlikely((val) == NULL)) {                                \
    fprintf(stderr, "NULL ERROR: %s:%d\n", __FILE__, __LINE__); \
    PyErr_Print();                                              \
    abort();                                                    \
  } else {                                                      \
  }

#define CHECK(cond)                                                     \
  if (unlikely(!(cond))) {                                              \
    fprintf(stderr, "DEBUG CHECK FAILED: %s:%d\n", __FILE__, __LINE__); \
    abort();                                                            \
  } else {                                                              \
  }

#ifdef TORCHDYNAMO_DEBUG

#define DEBUG_CHECK(cond) CHECK(cond)
#define DEBUG_NULL_CHECK(val) NULL_CHECK(val)
#define DEBUG_TRACE(msg, ...) \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__, __VA_ARGS__)
#define DEBUG_TRACE0(msg) \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__)

#else

#define DEBUG_CHECK(cond)
#define DEBUG_NULL_CHECK(val)
#define DEBUG_TRACE(msg, ...)
#define DEBUG_TRACE0(msg)

#endif

// Flag to just run a frame normally
#define SKIP_CODE ((void*)0x1)

static PyObject* noargs = NULL; /* cached empty tuple */
static PyObject* dotzerokey = NULL; /* ".0" */
static PyObject* guard_fail_hook = NULL;
static PyObject* guard_error_hook = NULL;

size_t extra_index = -1;

static Py_tss_t eval_frame_callback_key = Py_tss_NEEDS_INIT;

inline static PyObject* eval_frame_callback_get(void) {
  void* result = PyThread_tss_get(&eval_frame_callback_key);
  if (unlikely(result == NULL)) {
    Py_RETURN_NONE;
  } else {
    return (PyObject*)result;
  }
}

inline static void eval_frame_callback_set(PyObject* obj) {
  PyThread_tss_set(&eval_frame_callback_key, obj);
}

static void ignored(void* obj) {}
static PyObject* _custom_eval_frame_shim(
    PyThreadState* tstate,
    PyFrameObject* frame,
    int throw_flag);
static PyObject* _custom_eval_frame(
    PyThreadState* tstate,
    PyFrameObject* frame,
    int throw_flag,
    PyObject* callback);
#if PY_VERSION_HEX >= 0x03090000
static PyObject* custom_eval_frame_shim(
    PyThreadState* tstate,
    PyFrameObject* frame,
    int throw_flag) {
  return _custom_eval_frame_shim(tstate, frame, throw_flag);
}
#else
static PyObject* custom_eval_frame_shim(PyFrameObject* frame, int throw_flag) {
  PyThreadState* tstate = PyThreadState_GET();
  return _custom_eval_frame_shim(tstate, frame, throw_flag);
}
#endif

inline static PyObject* eval_frame_default(
    PyThreadState* tstate,
    PyFrameObject* frame,
    int throw_flag) {
#if PY_VERSION_HEX >= 0x03090000
  if (tstate == NULL) {
    tstate = PyThreadState_GET();
  }
  return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
#else
  return _PyEval_EvalFrameDefault(frame, throw_flag);
#endif
}

inline static void enable_eval_frame_shim(PyThreadState* tstate) {
#if PY_VERSION_HEX >= 0x03090000
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
      &custom_eval_frame_shim) {
    _PyInterpreterState_SetEvalFrameFunc(
        tstate->interp, &custom_eval_frame_shim);
  }
#else
  if (tstate->interp->eval_frame != &custom_eval_frame_shim) {
    // First call
    tstate->interp->eval_frame = &custom_eval_frame_shim;
  }
#endif
}

inline static void enable_eval_frame_default(PyThreadState* tstate) {
#if PY_VERSION_HEX >= 0x03090000
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
      &_PyEval_EvalFrameDefault) {
    _PyInterpreterState_SetEvalFrameFunc(
        tstate->interp, &_PyEval_EvalFrameDefault);
  }
#else
  if (tstate->interp->eval_frame != &_PyEval_EvalFrameDefault) {
    // First call
    tstate->interp->eval_frame = &_PyEval_EvalFrameDefault;
  }
#endif
}

static inline PyObject* call_callback(
    PyObject* callable,
    PyObject* frame,
    long cache_len) {
  PyObject* args = Py_BuildValue("(Ol)", frame, cache_len);
  NULL_CHECK(args);
  PyObject* result = PyObject_CallObject(callable, args);
  Py_DECREF(args);
  return result;
}

typedef struct cache_entry {
  // check the guards: lambda: <locals of user function>: bool
  PyObject* check_fn;
  // modified user bytecode (protected by check_fn's guards)
  PyCodeObject* code;
  // on a cache miss, linked list of next thing to try
  struct cache_entry* next;
} CacheEntry;

static CacheEntry* create_cache_entry(
    CacheEntry* next,
    PyObject* guarded_code) {
  CacheEntry* e = (CacheEntry*)malloc(sizeof(CacheEntry));
  DEBUG_NULL_CHECK(e);
  e->check_fn = PyObject_GetAttrString(guarded_code, "check_fn");
  NULL_CHECK(e->check_fn);
  e->code = (PyCodeObject*)PyObject_GetAttrString(guarded_code, "code");
  NULL_CHECK(e->code);
  e->next = next;
  return e;
}

static void destroy_cache_entry(CacheEntry* e) {
  if (e == NULL || e == SKIP_CODE) {
    return;
  }
  Py_XDECREF(e->check_fn);
  Py_XDECREF(e->code);
  destroy_cache_entry(e->next);
  free(e);
}

#ifdef TORCHDYNAMO_DEBUG
inline static const char* name(PyFrameObject* frame) {
  DEBUG_CHECK(PyUnicode_Check(frame->f_code->co_name));
  return PyUnicode_AsUTF8(frame->f_code->co_name);
}
#endif

static void call_guard_fail_hook(
    PyObject* hook,
    CacheEntry* e,
    PyObject* f_locals) {
  // call debugging logic when a guard fails
  PyObject* args = PyTuple_Pack(
      4,
      e->check_fn,
      e->code,
      f_locals,
      (e->next == NULL ? Py_True : Py_False));
  NULL_CHECK(args);
  PyObject* result = PyObject_CallObject(hook, args);
  NULL_CHECK(result);
  Py_DECREF(result);
  Py_DECREF(args);
}

static PyCodeObject* lookup(CacheEntry* e, PyObject* f_locals) {
  if (e == NULL) {
    return NULL;
  }
  PyObject* dotzero = PyDict_GetItem(f_locals, dotzerokey);
  PyObject* valid = NULL;
  if (unlikely(dotzero != NULL)) {
    // .0 is a special variable name used for implicit args
    PyObject* args = PyTuple_Pack(1, dotzero);
    NULL_CHECK(args);
    valid = PyObject_Call(e->check_fn, args, f_locals);
    Py_DECREF(args);
  } else {
    valid = PyObject_Call(e->check_fn, noargs, f_locals);
  }
  if (unlikely(valid == NULL)) {
    PyErr_Print();
    if (guard_error_hook != NULL) {
      call_guard_fail_hook(guard_error_hook, e, f_locals);
    }
    NULL_CHECK(valid);
  }
  Py_DECREF(valid);
  if (valid == Py_True) {
    return e->code;
  }
  if (unlikely(guard_fail_hook != NULL)) {
    call_guard_fail_hook(guard_fail_hook, e, f_locals);
  }
  return lookup(e->next, f_locals);
}

static long cache_size(CacheEntry* e) {
  if (e == NULL) {
    return 0;
  }
  return 1 + cache_size(e->next);
}

inline static CacheEntry* get_extra(PyCodeObject* code) {
  CacheEntry* extra = NULL;
  _PyCode_GetExtra((PyObject*)code, extra_index, (void*)&extra);
  return extra;
}

inline static void set_extra(PyCodeObject* code, CacheEntry* extra) {
  // TODO(jansel): would it be faster to bypass this?
  _PyCode_SetExtra((PyObject*)code, extra_index, extra);
}

inline static PyObject* eval_custom_code(
    PyThreadState* tstate,
    PyFrameObject* frame,
    PyCodeObject* code,
    int throw_flag) {
  Py_ssize_t ncells = 0;
  Py_ssize_t nfrees = 0;
  Py_ssize_t nlocals_new = code->co_nlocals;
  Py_ssize_t nlocals_old = frame->f_code->co_nlocals;

  if ((code->co_flags & CO_NOFREE) == 0) {
    ncells = PyTuple_GET_SIZE(code->co_cellvars);
    nfrees = PyTuple_GET_SIZE(code->co_freevars);
  }

  DEBUG_NULL_CHECK(tstate);
  DEBUG_NULL_CHECK(frame);
  DEBUG_NULL_CHECK(code);
  DEBUG_CHECK(ncells == PyTuple_GET_SIZE(frame->f_code->co_cellvars));
  DEBUG_CHECK(nfrees == PyTuple_GET_SIZE(frame->f_code->co_freevars));
  DEBUG_CHECK(nlocals_new >= nlocals_old);

  PyFrameObject* shadow = PyFrame_New(tstate, code, frame->f_globals, NULL);
  if (shadow == NULL) {
    return NULL;
  }

  PyObject** fastlocals_old = frame->f_localsplus;
  PyObject** fastlocals_new = shadow->f_localsplus;

  for (Py_ssize_t i = 0; i < nlocals_old; i++) {
    Py_XINCREF(fastlocals_old[i]);
    fastlocals_new[i] = fastlocals_old[i];
  }

  for (Py_ssize_t i = 0; i < ncells + nfrees; i++) {
    Py_XINCREF(fastlocals_old[nlocals_old + i]);
    fastlocals_new[nlocals_new + i] = fastlocals_old[nlocals_old + i];
  }

  PyObject* result = eval_frame_default(tstate, shadow, throw_flag);
  Py_DECREF(shadow);
  return result;
}

static PyObject* _custom_eval_frame_shim(
    PyThreadState* tstate,
    PyFrameObject* frame,
    int throw_flag) {
  // Shims logic into one of three states. Can probably be refactored into a
  // single func, later:
  //  - None: disables TorchDynamo
  //  - False: run-only mode (reuse existing compiles)
  //  - Python callable(): enables TorchDynamo
  PyObject* callback = eval_frame_callback_get();

  if (callback == Py_None) {
    return eval_frame_default(tstate, frame, throw_flag);
  }

  return _custom_eval_frame(tstate, frame, throw_flag, callback);
}

static PyObject* _custom_eval_frame(
    PyThreadState* tstate,
    PyFrameObject* frame,
    int throw_flag,
    PyObject* callback) {
  DEBUG_TRACE(
      "begin %s %s %i %i %i %i",
      name(frame),
      PyUnicode_AsUTF8(frame->f_code->co_filename),
      frame->f_lineno,
      frame->f_lasti,
      frame->f_iblock,
      frame->f_executing);
  CacheEntry* extra = get_extra(frame->f_code);
  if (extra == SKIP_CODE || (callback == Py_False && extra == NULL)) {
    DEBUG_TRACE("skip %s", name(frame));
    return eval_frame_default(tstate, frame, throw_flag);
  }

  // TODO(jansel): investigate directly using the "fast" representation
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
    DEBUG_TRACE("error %s", name(frame));
    return NULL;
  }

  // A callback of Py_False indicates "run only" mode, the cache is checked, but
  // we never compile.
  if (callback == Py_False) {
    DEBUG_TRACE("In run only mode %s", name(frame));
    PyCodeObject* cached_code = lookup(extra, frame->f_locals);
    if (cached_code != NULL) {
      // used cached version
      DEBUG_TRACE("cache hit %s", name(frame));
      return eval_custom_code(tstate, frame, cached_code, throw_flag);
    } else {
      DEBUG_TRACE("cache miss %s", name(frame));
      return eval_frame_default(tstate, frame, throw_flag);
    }
  }
  DEBUG_CHECK(PyDict_CheckExact(frame->f_locals));
  DEBUG_CHECK(PyDict_CheckExact(frame->f_globals));
  DEBUG_CHECK(PyDict_CheckExact(frame->f_builtins));

  // We don't run the current custom_eval_frame behavior for guards.
  // So we temporarily set the callback to Py_None to drive the correct behavior
  // in the shim.
  eval_frame_callback_set(Py_None);

  PyCodeObject* cached_code = lookup(extra, frame->f_locals);
  if (cached_code != NULL) {
    // used cached version
    DEBUG_TRACE("cache hit %s", name(frame));
    // Re-enable custom behavior
    eval_frame_callback_set(callback);
    return eval_custom_code(tstate, frame, cached_code, throw_flag);
  }
  // cache miss

  PyObject* result =
      call_callback(callback, (PyObject*)frame, cache_size(extra));
  if (result == NULL) {
    // internal exception, returning here will leak the exception into user code
    // this is useful for debugging -- but we dont want it to happen outside of
    // testing
    return NULL;
  } else if (result != Py_None) {
    DEBUG_TRACE("create cache %s", name(frame));
    extra = create_cache_entry(extra, result);
    Py_DECREF(result);
    set_extra(frame->f_code, extra);
    // Re-enable custom behavior
    eval_frame_callback_set(callback);
    return eval_custom_code(tstate, frame, extra->code, throw_flag);
  } else {
    DEBUG_TRACE("create skip %s", name(frame));
    Py_DECREF(result);
    destroy_cache_entry(extra);
    set_extra(frame->f_code, SKIP_CODE);
    // Re-enable custom behavior
    eval_frame_callback_set(callback);
    return eval_frame_default(tstate, frame, throw_flag);
  }
}

static int active_dynamo_threads = 0;

static PyObject* increment_working_threads(PyThreadState* tstate) {
  active_dynamo_threads = active_dynamo_threads + 1;
  if (active_dynamo_threads > 0) {
    enable_eval_frame_shim(tstate);
  }
  Py_RETURN_NONE;
}

static PyObject* decrement_working_threads(PyThreadState* tstate) {
  if (active_dynamo_threads > 0) {
    active_dynamo_threads = active_dynamo_threads - 1;
    if (active_dynamo_threads == 0) {
      enable_eval_frame_default(tstate);
    }
  }
  Py_RETURN_NONE;
}

static PyObject* set_eval_frame(PyObject* new_callback, PyThreadState* tstate) {
  // Change the eval frame callback and return the old one
  //  - None: disables TorchDynamo
  //  - False: run-only mode (reuse existing compiles)
  //  - Python callable(): enables TorchDynamo
  PyObject* old_callback = eval_frame_callback_get();

  // owned by caller
  Py_INCREF(old_callback);

  if (old_callback != Py_None && new_callback == Py_None) {
    decrement_working_threads(tstate);
  } else if (old_callback == Py_None && new_callback != Py_None) {
    increment_working_threads(tstate);
  }

  Py_INCREF(new_callback);
  Py_DECREF(old_callback);

  // Set thread local callback. This will drive behavior of our shim, if/when it
  // is installed.
  eval_frame_callback_set(new_callback);

  return old_callback;
}

static PyObject* set_eval_frame_py(PyObject* dummy, PyObject* args) {
  PyObject* callback = NULL;
  if (!PyArg_ParseTuple(args, "O:callback", &callback)) {
    DEBUG_TRACE0("arg error");
    return NULL;
  }
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
  return set_eval_frame(callback, PyThreadState_GET());
}

static PyObject* reset_code(PyObject* dummy, PyObject* args) {
  PyObject* code = NULL;
  if (!PyArg_ParseTuple(args, "O:code", &code)) {
    DEBUG_TRACE0("arg error");
    return NULL;
  }
  if (!PyCode_Check(code)) {
    DEBUG_TRACE0("arg error");
    PyErr_SetString(PyExc_TypeError, "expected a code object");
    return NULL;
  }

  destroy_cache_entry(get_extra((PyCodeObject*)code));
  set_extra((PyCodeObject*)code, NULL);
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

static PyObject* skip_code(PyObject* dummy, PyObject* args) {
  PyObject* obj = NULL;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  if (!PyCode_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "expected a code object");
    return NULL;
  }
  set_extra((PyCodeObject*)obj, SKIP_CODE);
  Py_RETURN_NONE;
}

static PyObject* set_guard_fail_hook(PyObject* dummy, PyObject* args) {
  PyObject* obj = NULL;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  Py_XDECREF(guard_fail_hook);
  if (obj == Py_None) {
    guard_fail_hook = NULL;
  } else {
    guard_fail_hook = obj;
    Py_INCREF(guard_fail_hook);
  }
  Py_RETURN_NONE;
}

static PyObject* set_guard_error_hook(PyObject* dummy, PyObject* args) {
  PyObject* obj = NULL;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  Py_XDECREF(guard_error_hook);
  if (obj == Py_None) {
    guard_error_hook = NULL;
  } else {
    guard_error_hook = obj;
    Py_INCREF(guard_error_hook);
  }
  Py_RETURN_NONE;
}

#else // python 3.11
#define PY311_RETURN_ERROR(name)                                          \
  static PyObject* name(PyObject* dummy, PyObject* args) {                \
    PyErr_SetString(PyExc_RuntimeError, "Python 3.11 not yet supported"); \
    return NULL;                                                          \
  }
PY311_RETURN_ERROR(set_eval_frame_py);
PY311_RETURN_ERROR(reset_code);
PY311_RETURN_ERROR(unsupported);
PY311_RETURN_ERROR(skip_code);
PY311_RETURN_ERROR(set_guard_fail_hook);
PY311_RETURN_ERROR(set_guard_error_hook);
#endif

static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame_py, METH_VARARGS, NULL},
    {"reset_code", reset_code, METH_VARARGS, NULL},
    {"unsupported", unsupported, METH_VARARGS, NULL},
    {"skip_code", skip_code, METH_VARARGS, NULL},
    {"set_guard_fail_hook", set_guard_fail_hook, METH_VARARGS, NULL},
    {"set_guard_error_hook", set_guard_error_hook, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.eval_frame",
    "Module containing hooks to override eval_frame",
    -1,
    _methods};

PyObject* torch_c_dynamo_eval_frame_init(void) {
#ifdef _PY_VERSION_OK
  extra_index = _PyEval_RequestCodeExtraIndex(ignored);

  int result = PyThread_tss_create(&eval_frame_callback_key);
  CHECK(result == 0);

  Py_INCREF(Py_None);
  eval_frame_callback_set(Py_None);

  noargs = PyTuple_New(0);
  dotzerokey = PyUnicode_InternFromString(".0");
#endif
  return PyModule_Create(&_module);
}

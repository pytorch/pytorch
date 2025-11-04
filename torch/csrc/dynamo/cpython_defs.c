#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/dynamo/debug_macros.h>

// Include CPython header files here (.c file) as MSVC C++ compiler cannot
// compile pycore_stackref.h. See PyTorch issue #160647
#if IS_PYTHON_3_14_PLUS && defined(_WIN32)
#define Py_BUILD_CORE
#include <internal/pycore_stackref.h>
#include <internal/pycore_genobject.h>
#include <internal/pycore_interpframe.h>
#undef Py_BUILD_CORE
#endif

#if IS_PYTHON_3_15_PLUS

const uint8_t* THP_PyOpcode_Caches = NULL;
int THP_PyOpcode_Caches_size = 0;

void THP_PyThreadState_PopFrame(
    PyThreadState* tstate,
    _PyInterpreterFrame* frame) {}
void THP_PyFrame_Clear(_PyInterpreterFrame* frame) {}

void init_THPCaches() {}

#else

#if IS_PYTHON_3_11_PLUS

#define Py_BUILD_CORE
#define NEED_OPCODE_TABLES // To get _PyOpcode_Deopt, _PyOpcode_Caches

#if IS_PYTHON_3_13_PLUS
#include <cpython/code.h> // To get PyUnstable_Code_GetFirstFree
#define NEED_OPCODE_METADATA
#include <internal/pycore_opcode_metadata.h>
#undef NEED_OPCODE_METADATA
#else
#include <internal/pycore_opcode.h>
#endif

#undef NEED_OPCODE_TABLES
#undef Py_BUILD_CORE

// As a simple way to reduce the impact of ABI changes on the CPython side, this
// check forces us to manually re-check that the function didn't change on the
// next major version
#if IS_PYTHON_3_15_PLUS
#error \
    "Please ensure that the functions below still match the CPython implementation for 3.15"
#endif

// e.g. COPY_FIELD(op, o, globals) becomes
// PY_XINCREF((o)->func_globals);
// (op)->func_globals = (o)->func_globals;
#define COPY_FIELD(f1, f2, field) \
  Py_XINCREF((f2)->func_##field); \
  (f1)->func_##field = (f2)->func_##field;

// Not actually copied from CPython, but loosely based on
// https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Objects/funcobject.c
// Makes a new PyFunctionObject copy of `o`, but with the code object fields
// determined from `code`.
// Ensure that all fields defined in the PyFunctionObject struct in
// https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Include/cpython/funcobject.h
// are accounted for.
PyFunctionObject* _PyFunction_CopyWithNewCode(
    PyFunctionObject* o,
    PyCodeObject* code) {
  PyFunctionObject* op = PyObject_GC_New(PyFunctionObject, &PyFunction_Type);
  if (op == NULL) {
    return NULL;
  }
  Py_XINCREF(code);
  op->func_code = (PyObject*)code;
  Py_XINCREF(code->co_name);
  op->func_name = code->co_name;
  Py_XINCREF(code->co_qualname);
  op->func_qualname = code->co_qualname;
  COPY_FIELD(op, o, globals);
  COPY_FIELD(op, o, builtins);
  COPY_FIELD(op, o, defaults);
  COPY_FIELD(op, o, kwdefaults);
  COPY_FIELD(op, o, closure);
  COPY_FIELD(op, o, doc);
  COPY_FIELD(op, o, dict);
  op->func_weakreflist = NULL;
  COPY_FIELD(op, o, module);
  COPY_FIELD(op, o, annotations);
#if IS_PYTHON_3_14_PLUS
  COPY_FIELD(op, o, annotate);
#endif
#if IS_PYTHON_3_12_PLUS
  COPY_FIELD(op, o, typeparams);
#endif
  op->vectorcall = o->vectorcall;
  op->func_version = o->func_version;
  PyObject_GC_Track(op);
  return op;
}

// From
// https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Objects/frameobject.c#L1020
PyFrameObject* THP_PyFrame_New_NoTrack(const PyCodeObject* code) {
  // DYNAMO: commented out
  // CALL_STAT_INC(frame_objects_created);
  int slots = code->co_nlocalsplus + code->co_stacksize;
  PyFrameObject* f = PyObject_GC_NewVar(PyFrameObject, &PyFrame_Type, slots);
  if (f == NULL) {
    return NULL;
  }
  f->f_back = NULL;
  f->f_trace = NULL;
  f->f_trace_lines = 1;
  f->f_trace_opcodes = 0;
#if IS_PYTHON_3_13_PLUS
  f->f_extra_locals = NULL;
#else
  f->f_fast_as_locals = 0;
#endif
  f->f_lineno = 0;
#if IS_PYTHON_3_14_PLUS
  f->f_locals_cache = NULL;
  f->f_overwritten_fast_locals = NULL;
#endif
  return f;
}

#if IS_PYTHON_3_14_PLUS

// From
// https://github.com/python/cpython/blob/8b3f9ae2ca55b2cc7edc097321cc10d7c2fdbb98/Python/frame.c#L21
PyFrameObject* THP_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame* frame) {
  CHECK(frame->frame_obj == NULL);
  PyObject* exc = PyErr_GetRaisedException();

  PyFrameObject* f = THP_PyFrame_New_NoTrack(F_CODE(frame));
  if (f == NULL) {
    Py_XDECREF(exc);
    return NULL;
  }
  PyErr_SetRaisedException(exc);

  // GH-97002: There was a time when a frame object could be created when we
  // are allocating the new frame object f above, so frame->frame_obj would
  // be assigned already. That path does not exist anymore. We won't call any
  // Python code in this function and garbage collection will not run.
  // Notice that _PyFrame_New_NoTrack() can potentially raise a MemoryError,
  // but it won't allocate a traceback until the frame unwinds, so we are safe
  // here.
  assert(frame->frame_obj == NULL);
  assert(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
  f->f_frame = frame;
  frame->frame_obj = f;
  return f;
}

#else

// From
// https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Python/frame.c#L27
PyFrameObject* THP_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame* frame) {
  CHECK(frame->frame_obj == NULL);
  PyObject *error_type = NULL, *error_value = NULL, *error_traceback = NULL;
  PyErr_Fetch(&error_type, &error_value, &error_traceback);

  PyFrameObject* f = THP_PyFrame_New_NoTrack(F_CODE(frame));
  if (f == NULL) {
    Py_XDECREF(error_type);
    Py_XDECREF(error_value);
    Py_XDECREF(error_traceback);
    return NULL;
  }
  PyErr_Restore(error_type, error_value, error_traceback);
  if (frame->frame_obj) {
    // GH-97002: How did we get into this horrible situation? Most likely,
    // allocating f triggered a GC collection, which ran some code that
    // *also* created the same frame... while we were in the middle of
    // creating it! See test_sneaky_frame_object in test_frame.py for a
    // concrete example.
    //
    // Regardless, just throw f away and use that frame instead, since it's
    // already been exposed to user code. It's actually a bit tricky to do
    // this, since we aren't backed by a real _PyInterpreterFrame anymore.
    // Just pretend that we have an owned, cleared frame so frame_dealloc
    // doesn't make the situation worse:
    f->f_frame = (_PyInterpreterFrame*)f->_f_frame_data;
    f->f_frame->owner = FRAME_CLEARED;
    f->f_frame->frame_obj = f;
    Py_DECREF(f);
    return frame->frame_obj;
  }
  CHECK(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
  CHECK(frame->owner != FRAME_CLEARED);
  f->f_frame = frame;
  frame->frame_obj = f;
  return f;
}

#endif

// From
// https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Include/internal/pycore_frame.h#L163
static inline PyFrameObject* THP_PyFrame_GetFrameObject(
    _PyInterpreterFrame* frame) {
  CHECK(!_PyFrame_IsIncomplete(frame));
  PyFrameObject* res = frame->frame_obj;
  if (res != NULL) {
    return res;
  }
  return THP_PyFrame_MakeAndSetFrameObject(frame);
}

#if IS_PYTHON_3_14_PLUS

static void THP_take_ownership(PyFrameObject* f, _PyInterpreterFrame* frame) {
  Py_BEGIN_CRITICAL_SECTION(f);
  CHECK(frame->owner < FRAME_OWNED_BY_INTERPRETER);
  CHECK(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
  _PyInterpreterFrame* new_frame = (_PyInterpreterFrame*)f->_f_frame_data;
  _PyFrame_Copy(frame, new_frame);
  // _PyFrame_Copy takes the reference to the executable,
  // so we need to restore it.
  frame->f_executable = PyStackRef_DUP(new_frame->f_executable);
  f->f_frame = new_frame;
  new_frame->owner = FRAME_OWNED_BY_FRAME_OBJECT;
  if (_PyFrame_IsIncomplete(new_frame)) {
    // This may be a newly-created generator or coroutine frame. Since it's
    // dead anyways, just pretend that the first RESUME ran:
    PyCodeObject* code = F_CODE(new_frame);
    new_frame->instr_ptr =
        _PyFrame_GetBytecode(new_frame) + code->_co_firsttraceable + 1;
  }
  CHECK(!_PyFrame_IsIncomplete(new_frame));
  CHECK(f->f_back == NULL);
  _PyInterpreterFrame* prev = _PyFrame_GetFirstComplete(frame->previous);
  if (prev) {
    CHECK(prev->owner < FRAME_OWNED_BY_INTERPRETER);
    PyObject* exc = PyErr_GetRaisedException();
    /* Link PyFrameObjects.f_back and remove link through
     * _PyInterpreterFrame.previous */
    PyFrameObject* back = THP_PyFrame_GetFrameObject(prev);
    if (back == NULL) {
      /* Memory error here. */
      assert(PyErr_ExceptionMatches(PyExc_MemoryError));
      /* Nothing we can do about it */
      PyErr_Clear();
    } else {
      f->f_back = (PyFrameObject*)Py_NewRef(back);
    }
    PyErr_SetRaisedException(exc);
  }
  if (!_PyObject_GC_IS_TRACKED((PyObject*)f)) {
    _PyObject_GC_TRACK((PyObject*)f);
  }
  Py_END_CRITICAL_SECTION();
}

#else

// From
// https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Python/frame.c#L79
static void THP_take_ownership(PyFrameObject* f, _PyInterpreterFrame* frame) {
  CHECK(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
  CHECK(frame->owner != FRAME_CLEARED);
  Py_ssize_t size = ((char*)&frame->localsplus[frame->stacktop]) - (char*)frame;
  memcpy((_PyInterpreterFrame*)f->_f_frame_data, frame, size);
  frame = (_PyInterpreterFrame*)f->_f_frame_data;
  f->f_frame = frame;
  frame->owner = FRAME_OWNED_BY_FRAME_OBJECT;
  if (_PyFrame_IsIncomplete(frame)) {
    // This may be a newly-created generator or coroutine frame. Since it's
    // dead anyways, just pretend that the first RESUME ran:
    PyCodeObject* code = F_CODE(frame);
    PREV_INSTR(frame) = _PyCode_CODE(code) + code->_co_firsttraceable;
  }
  CHECK(!_PyFrame_IsIncomplete(frame));
  CHECK(f->f_back == NULL);
  _PyInterpreterFrame* prev = frame->previous;
  while (prev && _PyFrame_IsIncomplete(prev)) {
    prev = prev->previous;
  }
  if (prev) {
    /* Link PyFrameObjects.f_back and remove link through
     * _PyInterpreterFrame.previous */
    PyFrameObject* back = THP_PyFrame_GetFrameObject(prev);
    if (back == NULL) {
      /* Memory error here. */
      CHECK(PyErr_ExceptionMatches(PyExc_MemoryError));
      /* Nothing we can do about it */
      PyErr_Clear();
    } else {
      f->f_back = (PyFrameObject*)Py_NewRef(back);
    }
    frame->previous = NULL;
  }
  // DYNAMO: use public GC functions instead of internal ones
  if (!PyObject_GC_IsTracked((PyObject*)f)) {
    PyObject_GC_Track((PyObject*)f);
  }
}

#endif

#if IS_PYTHON_3_14_PLUS

void THP_PyFrame_ClearLocals(_PyInterpreterFrame* frame) {
  CHECK(frame->stackpointer != NULL);
  _PyStackRef* sp = frame->stackpointer;
  _PyStackRef* locals = frame->localsplus;
  frame->stackpointer = locals;
  while (sp > locals) {
    sp--;
    PyStackRef_XCLOSE(*sp);
  }
  Py_CLEAR(frame->f_locals);
}

// From
// https://github.com/python/cpython/blob/8b3f9ae2ca55b2cc7edc097321cc10d7c2fdbb98/Python/frame.c#L107
void THP_PyFrame_Clear(_PyInterpreterFrame* frame) {
  /* It is the responsibility of the owning generator/coroutine
   * to have cleared the enclosing generator, if any. */
  CHECK(
      frame->owner != FRAME_OWNED_BY_GENERATOR ||
      _PyGen_GetGeneratorFromFrame(frame)->gi_frame_state == FRAME_CLEARED);
  // GH-99729: Clearing this frame can expose the stack (via finalizers). It's
  // crucial that this frame has been unlinked, and is no longer visible:
  CHECK(_PyThreadState_GET()->current_frame != frame);
  if (frame->frame_obj) {
    PyFrameObject* f = frame->frame_obj;
    frame->frame_obj = NULL;
    if (!_PyObject_IsUniquelyReferenced((PyObject*)f)) {
      THP_take_ownership(f, frame);
      Py_DECREF(f);
      return;
    }
    Py_DECREF(f);
  }
  THP_PyFrame_ClearLocals(frame);
  PyStackRef_CLEAR(frame->f_funcobj);
}

#else

// From
// https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Python/frame.c#L120
void THP_PyFrame_Clear(_PyInterpreterFrame* frame) {
  /* It is the responsibility of the owning generator/coroutine
   * to have cleared the enclosing generator, if any. */
  CHECK(
      frame->owner != FRAME_OWNED_BY_GENERATOR ||
      _PyFrame_GetGenerator(frame)->gi_frame_state == FRAME_CLEARED);
  // GH-99729: Clearing this frame can expose the stack (via finalizers). It's
  // crucial that this frame has been unlinked, and is no longer visible:
#if IS_PYTHON_3_13_PLUS
  CHECK(_PyThreadState_GET()->current_frame != frame);
#else
  CHECK(_PyThreadState_GET()->cframe->current_frame != frame);
#endif
  if (frame->frame_obj) {
    PyFrameObject* f = frame->frame_obj;
    frame->frame_obj = NULL;
    if (Py_REFCNT(f) > 1) {
      THP_take_ownership(f, frame);
      Py_DECREF(f);
      return;
    }
    Py_DECREF(f);
  }
  CHECK(frame->stacktop >= 0);
  for (int i = 0; i < frame->stacktop; i++) {
    Py_XDECREF(frame->localsplus[i]);
  }
  Py_XDECREF(frame->frame_obj);
  Py_XDECREF(frame->f_locals);
// DYNAMO: additional field for 3.12
#if IS_PYTHON_3_12_PLUS
  Py_DECREF(frame->f_funcobj);
#else
  Py_DECREF(frame->f_func);
#endif
  Py_DECREF(F_CODE(frame));
}

#endif

// https://github.com/python/cpython/blob/fad48ea1816be3125ea51edcdfe2f999d6ade796/Objects/obmalloc.c#L635
void* THP_PyObject_VirtualAlloc(size_t size) {
  PyObjectArenaAllocator arena;
  PyObject_GetArenaAllocator(&arena);
  return arena.alloc(arena.ctx, size);
}

// https://github.com/python/cpython/blob/fad48ea1816be3125ea51edcdfe2f999d6ade796/Objects/obmalloc.c#L641
void THP_PyObject_VirtualFree(void* obj, size_t size) {
  PyObjectArenaAllocator arena;
  PyObject_GetArenaAllocator(&arena);
  arena.free(arena.ctx, obj, size);
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L728
static _PyStackChunk* allocate_chunk(
    int size_in_bytes,
    _PyStackChunk* previous) {
  CHECK(size_in_bytes % sizeof(PyObject**) == 0);
  _PyStackChunk* res = THP_PyObject_VirtualAlloc(size_in_bytes);
  if (res == NULL) {
    return NULL;
  }
  res->previous = previous;
  res->size = size_in_bytes;
  res->top = 0;
  return res;
}

#define DATA_STACK_CHUNK_SIZE (16 * 1024)
#define MINIMUM_OVERHEAD 1000

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L2182
static PyObject** push_chunk(PyThreadState* tstate, int size) {
  int allocate_size = DATA_STACK_CHUNK_SIZE;
  while (allocate_size < (int)sizeof(PyObject*) * (size + MINIMUM_OVERHEAD)) {
    allocate_size *= 2;
  }
  _PyStackChunk* new = allocate_chunk(allocate_size, tstate->datastack_chunk);
  if (new == NULL) {
    return NULL;
  }
  if (tstate->datastack_chunk) {
    tstate->datastack_chunk->top =
        tstate->datastack_top - &tstate->datastack_chunk->data[0];
  }
  tstate->datastack_chunk = new;
  tstate->datastack_limit = (PyObject**)(((char*)new) + allocate_size);
  // When new is the "root" chunk (i.e. new->previous == NULL), we can keep
  // _PyThreadState_PopFrame from freeing it later by "skipping" over the
  // first element:
  PyObject** res = &new->data[new->previous == NULL];
  tstate->datastack_top = res + size;
  return res;
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Include/internal/pycore_frame.h#L199
static inline bool THP_PyThreadState_HasStackSpace(
    PyThreadState* tstate,
    size_t size) {
  CHECK(
      (tstate->datastack_top == NULL && tstate->datastack_limit == NULL) ||
      (tstate->datastack_top != NULL && tstate->datastack_limit != NULL));
  return tstate->datastack_top != NULL &&
      size < (size_t)(tstate->datastack_limit - tstate->datastack_top);
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L2207
_PyInterpreterFrame* THP_PyThreadState_BumpFramePointerSlow(
    PyThreadState* tstate,
    size_t size) {
  if (THP_PyThreadState_HasStackSpace(tstate, size)) {
    _PyInterpreterFrame* res = (_PyInterpreterFrame*)tstate->datastack_top;
    tstate->datastack_top += size;
    return res;
  }
  if (size > INT_MAX / 2) {
    PyErr_NoMemory();
    return NULL;
  }
  return (_PyInterpreterFrame*)push_chunk(tstate, (int)size);
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L2222
void THP_PyThreadState_PopFrame(
    PyThreadState* tstate,
    _PyInterpreterFrame* frame) {
  CHECK(tstate->datastack_chunk);
  PyObject** base = (PyObject**)frame;
  if (base == &tstate->datastack_chunk->data[0]) {
    _PyStackChunk* chunk = tstate->datastack_chunk;
    _PyStackChunk* previous = chunk->previous;
    // push_chunk ensures that the root chunk is never popped:
    CHECK(previous);
    tstate->datastack_top = &previous->data[previous->top];
    tstate->datastack_chunk = previous;
    THP_PyObject_VirtualFree(chunk, chunk->size);
    tstate->datastack_limit = (PyObject**)(((char*)previous) + previous->size);
  } else {
    CHECK(tstate->datastack_top);
    CHECK(tstate->datastack_top >= base);
    tstate->datastack_top = base;
  }
}

#endif

const uint8_t* THP_PyOpcode_Caches = NULL;
int THP_PyOpcode_Caches_size = 0;
void init_THPCaches() {
#if IS_PYTHON_3_11_PLUS
  THP_PyOpcode_Caches = _PyOpcode_Caches;
  THP_PyOpcode_Caches_size = sizeof(_PyOpcode_Caches) / sizeof(uint8_t);
#endif
}

#endif // IS_PYTHON_3_15_PLUS

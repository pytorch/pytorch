#include <torch/csrc/dynamo/cpython_defs.h>

#ifdef _WIN32
#define unlikely(x) (x)
#else
#define unlikely(x) __builtin_expect((x), 0)
#endif

#define CHECK(cond)                                                     \
  if (unlikely(!(cond))) {                                              \
    fprintf(stderr, "DEBUG CHECK FAILED: %s:%d\n", __FILE__, __LINE__); \
    abort();                                                            \
  } else {                                                              \
  }

// NOTE: all `assert`s below are converted to `CHECK`s

#if IS_PYTHON_3_11_PLUS

// Problem in CPython includes when mixing core and non-core build
// The fix was not backported to 3.12 so this is needed here
// https://github.com/python/cpython/issues/105268
#if IS_PYTHON_3_12_PLUS
#undef _PyGC_FINALIZED
#endif

#define Py_BUILD_CORE
#include <internal/pycore_pystate.h>
#define NEED_OPCODE_TABLES // To get _PyOpcode_Deopt, _PyOpcode_Caches
#include <internal/pycore_opcode.h>
#undef NEED_OPCODE_TABLES
#undef Py_BUILD_CORE
#include <internal/pycore_frame.h>

// As a simple way to reduce the impact of ABI changes on the CPython side, this check forces
// us to manually re-check that the function didn't change on the next major version
#if PY_VERSION_HEX >= 0x030D0000 // 3.13
#error "Please ensure that the functions below still match the CPython implementation for 3.13"
#endif

// https://github.com/python/cpython/blob/a7715ccfba5b86ab09f86ec56ac3755c93b46b48/Objects/frameobject.c#L1079
static int
THP_PyFrame_OpAlreadyRan(_PyInterpreterFrame *frame, int opcode, int oparg)
{
    // This only works when opcode is a non-quickened form:
    CHECK(_PyOpcode_Deopt[opcode] == opcode);
    int check_oparg = 0;
    for (_Py_CODEUNIT *instruction = _PyCode_CODE(frame->f_code);
         instruction < frame->prev_instr; instruction++)
    {
        int check_opcode = _PyOpcode_Deopt[_Py_OPCODE(*instruction)];
        check_oparg |= _Py_OPARG(*instruction);
        if (check_opcode == opcode && check_oparg == oparg) {
            return 1;
        }
        if (check_opcode == EXTENDED_ARG) {
            check_oparg <<= 8;
        }
        else {
            check_oparg = 0;
        }
        instruction += _PyOpcode_Caches[check_opcode];
    }
    return 0;
}

#if IS_PYTHON_3_12_PLUS

// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1136
// Initialize frame free variables if needed
// free_vars_copied argument added in order to let caller know that the COPY_FREE_VARS
// codepath occurred.
static void
frame_init_get_vars(_PyInterpreterFrame *frame, int *free_vars_copied)
{
    // COPY_FREE_VARS has no quickened forms, so no need to use _PyOpcode_Deopt
    // here:
    PyCodeObject *co = frame->f_code;
    int lasti = _PyInterpreterFrame_LASTI(frame);
    if (!(lasti < 0 && _PyCode_CODE(co)->op.code == COPY_FREE_VARS
          && PyFunction_Check(frame->f_funcobj)))
    {
        /* Free vars are initialized */
        return;
    }

    /* Free vars have not been initialized -- Do that */
    PyObject *closure = ((PyFunctionObject *)frame->f_funcobj)->func_closure;
    int offset = PyCode_GetFirstFree(co);
    for (int i = 0; i < co->co_nfreevars; ++i) {
        PyObject *o = PyTuple_GET_ITEM(closure, i);
        frame->localsplus[offset + i] = Py_NewRef(o);
    }
    // COPY_FREE_VARS doesn't have inline CACHEs, either:
    frame->prev_instr = _PyCode_CODE(frame->f_code);

    *free_vars_copied = 1;
}

// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1162
static int
frame_get_var(_PyInterpreterFrame *frame, PyCodeObject *co, int i,
              PyObject **pvalue)
{
    _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);

    /* If the namespace is unoptimized, then one of the
       following cases applies:
       1. It does not contain free variables, because it
          uses import * or is a top-level namespace.
       2. It is a class namespace.
       We don't want to accidentally copy free variables
       into the locals dict used by the class.
    */
    if (kind & CO_FAST_FREE && !(co->co_flags & CO_OPTIMIZED)) {
        return 0;
    }

    PyObject *value = frame->localsplus[i];
    if (frame->stacktop) {
        if (kind & CO_FAST_FREE) {
            // The cell was set by COPY_FREE_VARS.
            CHECK(value != NULL && PyCell_Check(value));
            value = PyCell_GET(value);
        }
        else if (kind & CO_FAST_CELL) {
            // Note that no *_DEREF ops can happen before MAKE_CELL
            // executes.  So there's no need to duplicate the work
            // that MAKE_CELL would otherwise do later, if it hasn't
            // run yet.
            if (value != NULL) {
                if (PyCell_Check(value) &&
                        THP_PyFrame_OpAlreadyRan(frame, MAKE_CELL, i)) {
                    // (likely) MAKE_CELL must have executed already.
                    value = PyCell_GET(value);
                }
                // (likely) Otherwise it it is an arg (kind & CO_FAST_LOCAL),
                // with the initial value set when the frame was created...
                // (unlikely) ...or it was set to some initial value by
                // an earlier call to PyFrame_LocalsToFast().
            }
        }
    }
    else {
        CHECK(value == NULL);
    }
    *pvalue = value;
    return 1;
}

// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1213
static PyObject *
THP_PyFrame_GetLocals(_PyInterpreterFrame *frame, int include_hidden, int *free_vars_copied)
{
    /* Merge fast locals into f->f_locals */
    PyObject *locals = frame->f_locals;
    if (locals == NULL) {
        locals = frame->f_locals = PyDict_New();
        if (locals == NULL) {
            return NULL;
        }
    }
    PyObject *hidden = NULL;

    /* If include_hidden, "hidden" fast locals (from inlined comprehensions in
       module/class scopes) will be included in the returned dict, but not in
       frame->f_locals; the returned dict will be a modified copy. Non-hidden
       locals will still be updated in frame->f_locals. */
    if (include_hidden) {
        hidden = PyDict_New();
        if (hidden == NULL) {
            return NULL;
        }
    }

    frame_init_get_vars(frame, free_vars_copied);

    PyCodeObject *co = frame->f_code;
    for (int i = 0; i < co->co_nlocalsplus; i++) {
        PyObject *value;  // borrowed reference
        if (!frame_get_var(frame, co, i, &value)) {
            continue;
        }

        PyObject *name = PyTuple_GET_ITEM(co->co_localsplusnames, i);
        _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);
        if (kind & CO_FAST_HIDDEN) {
            if (include_hidden && value != NULL) {
                if (PyObject_SetItem(hidden, name, value) != 0) {
                    goto error;
                }
            }
            continue;
        }
        if (value == NULL) {
            if (PyObject_DelItem(locals, name) != 0) {
                if (PyErr_ExceptionMatches(PyExc_KeyError)) {
                    PyErr_Clear();
                }
                else {
                    goto error;
                }
            }
        }
        else {
            if (PyObject_SetItem(locals, name, value) != 0) {
                goto error;
            }
        }
    }

    if (include_hidden && PyDict_Size(hidden)) {
        PyObject *innerlocals = PyDict_New();
        if (innerlocals == NULL) {
            goto error;
        }
        if (PyDict_Merge(innerlocals, locals, 1) != 0) {
            Py_DECREF(innerlocals);
            goto error;
        }
        if (PyDict_Merge(innerlocals, hidden, 1) != 0) {
            Py_DECREF(innerlocals);
            goto error;
        }
        locals = innerlocals;
    }
    else {
        Py_INCREF(locals);
    }
    Py_CLEAR(hidden);

    return locals;

  error:
    Py_XDECREF(hidden);
    return NULL;
}

// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1301
int
THP_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame, int *free_vars_copied)
{
    PyObject *locals = THP_PyFrame_GetLocals(frame, 0, free_vars_copied);
    if (locals == NULL) {
        return -1;
    }
    Py_DECREF(locals);
    return 0;
}

#else

// https://github.com/python/cpython/blob/a7715ccfba5b86ab09f86ec56ac3755c93b46b48/Objects/frameobject.c#L1182
// free_vars_copied argument added in order to let caller know that the COPY_FREE_VARS
// codepath occurred.
int
THP_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame, int *free_vars_copied) {
    /* Merge fast locals into f->f_locals */
    PyObject *locals = NULL;
    PyObject **fast = NULL;
    PyCodeObject *co = NULL;
    locals = frame->f_locals;
    if (locals == NULL) {
        locals = frame->f_locals = PyDict_New();
        if (locals == NULL)
            return -1;
    }
    co = frame->f_code;
    fast = _PyFrame_GetLocalsArray(frame);
    // COPY_FREE_VARS has no quickened forms, so no need to use _PyOpcode_Deopt
    // here:
    int lasti = _PyInterpreterFrame_LASTI(frame);
    if (lasti < 0 && _Py_OPCODE(_PyCode_CODE(co)[0]) == COPY_FREE_VARS) {
        /* Free vars have not been initialized -- Do that */
        PyCodeObject *co = frame->f_code;
        PyObject *closure = frame->f_func->func_closure;
        int offset = co->co_nlocals + co->co_nplaincellvars;
        for (int i = 0; i < co->co_nfreevars; ++i) {
            PyObject *o = PyTuple_GET_ITEM(closure, i);
            Py_INCREF(o);
            frame->localsplus[offset + i] = o;
        }
        // COPY_FREE_VARS doesn't have inline CACHEs, either:
        frame->prev_instr = _PyCode_CODE(frame->f_code);

        *free_vars_copied = 1;
    }
    for (int i = 0; i < co->co_nlocalsplus; i++) {
        _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);

        /* If the namespace is unoptimized, then one of the
           following cases applies:
           1. It does not contain free variables, because it
              uses import * or is a top-level namespace.
           2. It is a class namespace.
           We don't want to accidentally copy free variables
           into the locals dict used by the class.
        */
        if (kind & CO_FAST_FREE && !(co->co_flags & CO_OPTIMIZED)) {
            continue;
        }

        PyObject *name = PyTuple_GET_ITEM(co->co_localsplusnames, i);
        PyObject *value = fast[i];
        if (frame->stacktop) {
            if (kind & CO_FAST_FREE) {
                // The cell was set by COPY_FREE_VARS.
                CHECK(value != NULL && PyCell_Check(value));
                value = PyCell_GET(value);
            }
            else if (kind & CO_FAST_CELL) {
                // Note that no *_DEREF ops can happen before MAKE_CELL
                // executes.  So there's no need to duplicate the work
                // that MAKE_CELL would otherwise do later, if it hasn't
                // run yet.
                if (value != NULL) {
                    if (PyCell_Check(value) &&
                            THP_PyFrame_OpAlreadyRan(frame, MAKE_CELL, i)) {
                        // (likely) MAKE_CELL must have executed already.
                        value = PyCell_GET(value);
                    }
                    // (likely) Otherwise it it is an arg (kind & CO_FAST_LOCAL),
                    // with the initial value set when the frame was created...
                    // (unlikely) ...or it was set to some initial value by
                    // an earlier call to PyFrame_LocalsToFast().
                }
            }
        }
        else {
            CHECK(value == NULL);
        }
        if (value == NULL) {
            if (PyObject_DelItem(locals, name) != 0) {
                if (PyErr_ExceptionMatches(PyExc_KeyError)) {
                    PyErr_Clear();
                }
                else {
                    return -1;
                }
            }
        }
        else {
            if (PyObject_SetItem(locals, name, value) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

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
PyFunctionObject *
_PyFunction_CopyWithNewCode(PyFunctionObject *o, PyCodeObject* code)
{
  PyFunctionObject *op = PyObject_GC_New(PyFunctionObject, &PyFunction_Type);
  if (op == NULL) {
    return NULL;
  }
  Py_XINCREF(code);
  op->func_code = (PyObject *) code;
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
  #if IS_PYTHON_3_12_PLUS
  COPY_FIELD(op, o, typeparams);
  #endif
  op->vectorcall = o->vectorcall;
  op->func_version = o->func_version;
  PyObject_GC_Track(op);
  return op;
}

// From https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Objects/frameobject.c#L1020
PyFrameObject*
THP_PyFrame_New_NoTrack(const PyCodeObject *code)
{
    // DYNAMO: commented out
    // CALL_STAT_INC(frame_objects_created);
    int slots = code->co_nlocalsplus + code->co_stacksize;
    PyFrameObject *f = PyObject_GC_NewVar(PyFrameObject, &PyFrame_Type, slots);
    if (f == NULL) {
        return NULL;
    }
    f->f_back = NULL;
    f->f_trace = NULL;
    f->f_trace_lines = 1;
    f->f_trace_opcodes = 0;
    f->f_fast_as_locals = 0;
    f->f_lineno = 0;
    return f;
}

// From https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Python/frame.c#L27
PyFrameObject *
THP_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame *frame)
{
    CHECK(frame->frame_obj == NULL);
    PyObject *error_type = NULL, *error_value = NULL, *error_traceback = NULL;
    PyErr_Fetch(&error_type, &error_value, &error_traceback);

    PyFrameObject *f = THP_PyFrame_New_NoTrack(frame->f_code);
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
        f->f_frame = (_PyInterpreterFrame *)f->_f_frame_data;
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

// From https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Include/internal/pycore_frame.h#L163
static inline PyFrameObject *
THP_PyFrame_GetFrameObject(_PyInterpreterFrame *frame)
{

    CHECK(!_PyFrame_IsIncomplete(frame));
    PyFrameObject *res =  frame->frame_obj;
    if (res != NULL) {
        return res;
    }
    return THP_PyFrame_MakeAndSetFrameObject(frame);
}

// From https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Python/frame.c#L79
static void
THP_take_ownership(PyFrameObject *f, _PyInterpreterFrame *frame)
{
    CHECK(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
    CHECK(frame->owner != FRAME_CLEARED);
    Py_ssize_t size = ((char*)&frame->localsplus[frame->stacktop]) - (char *)frame;
    memcpy((_PyInterpreterFrame *)f->_f_frame_data, frame, size);
    frame = (_PyInterpreterFrame *)f->_f_frame_data;
    f->f_frame = frame;
    frame->owner = FRAME_OWNED_BY_FRAME_OBJECT;
    if (_PyFrame_IsIncomplete(frame)) {
        // This may be a newly-created generator or coroutine frame. Since it's
        // dead anyways, just pretend that the first RESUME ran:
        PyCodeObject *code = frame->f_code;
        frame->prev_instr = _PyCode_CODE(code) + code->_co_firsttraceable;
    }
    CHECK(!_PyFrame_IsIncomplete(frame));
    CHECK(f->f_back == NULL);
    _PyInterpreterFrame *prev = frame->previous;
    while (prev && _PyFrame_IsIncomplete(prev)) {
        prev = prev->previous;
    }
    if (prev) {
        /* Link PyFrameObjects.f_back and remove link through _PyInterpreterFrame.previous */
        PyFrameObject *back = THP_PyFrame_GetFrameObject(prev);
        if (back == NULL) {
            /* Memory error here. */
            CHECK(PyErr_ExceptionMatches(PyExc_MemoryError));
            /* Nothing we can do about it */
            PyErr_Clear();
        }
        else {
            f->f_back = (PyFrameObject *)Py_NewRef(back);
        }
        frame->previous = NULL;
    }
    // DYNAMO: use public GC functions instead of internal ones
    if (!PyObject_GC_IsTracked((PyObject *) f)) {
        PyObject_GC_Track((PyObject *) f);
    }
}

// From https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Python/frame.c#L120
void
THP_PyFrame_Clear(_PyInterpreterFrame *frame)
{
    /* It is the responsibility of the owning generator/coroutine
     * to have cleared the enclosing generator, if any. */
    CHECK(frame->owner != FRAME_OWNED_BY_GENERATOR ||
        _PyFrame_GetGenerator(frame)->gi_frame_state == FRAME_CLEARED);
    // GH-99729: Clearing this frame can expose the stack (via finalizers). It's
    // crucial that this frame has been unlinked, and is no longer visible:
    CHECK(_PyThreadState_GET()->cframe->current_frame != frame);
    if (frame->frame_obj) {
        PyFrameObject *f = frame->frame_obj;
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
    Py_DECREF(frame->f_code);
}

// https://github.com/python/cpython/blob/fad48ea1816be3125ea51edcdfe2f999d6ade796/Objects/obmalloc.c#L635
void *
THP_PyObject_VirtualAlloc(size_t size)
{
    PyObjectArenaAllocator arena;
    PyObject_GetArenaAllocator(&arena);
    return arena.alloc(arena.ctx, size);
}

// https://github.com/python/cpython/blob/fad48ea1816be3125ea51edcdfe2f999d6ade796/Objects/obmalloc.c#L641
void
THP_PyObject_VirtualFree(void *obj, size_t size)
{
    PyObjectArenaAllocator arena;
    PyObject_GetArenaAllocator(&arena);
    return arena.free(arena.ctx, obj, size);
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L728
static _PyStackChunk*
allocate_chunk(int size_in_bytes, _PyStackChunk* previous)
{
    CHECK(size_in_bytes % sizeof(PyObject **) == 0);
    _PyStackChunk *res = THP_PyObject_VirtualAlloc(size_in_bytes);
    if (res == NULL) {
        return NULL;
    }
    res->previous = previous;
    res->size = size_in_bytes;
    res->top = 0;
    return res;
}

#define DATA_STACK_CHUNK_SIZE (16*1024)
#define MINIMUM_OVERHEAD 1000

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L2182
static PyObject **
push_chunk(PyThreadState *tstate, int size)
{
    int allocate_size = DATA_STACK_CHUNK_SIZE;
    while (allocate_size < (int)sizeof(PyObject*)*(size + MINIMUM_OVERHEAD)) {
        allocate_size *= 2;
    }
    _PyStackChunk *new = allocate_chunk(allocate_size, tstate->datastack_chunk);
    if (new == NULL) {
        return NULL;
    }
    if (tstate->datastack_chunk) {
        tstate->datastack_chunk->top = tstate->datastack_top -
                                       &tstate->datastack_chunk->data[0];
    }
    tstate->datastack_chunk = new;
    tstate->datastack_limit = (PyObject **)(((char *)new) + allocate_size);
    // When new is the "root" chunk (i.e. new->previous == NULL), we can keep
    // _PyThreadState_PopFrame from freeing it later by "skipping" over the
    // first element:
    PyObject **res = &new->data[new->previous == NULL];
    tstate->datastack_top = res + size;
    return res;
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Include/internal/pycore_frame.h#L199
static inline bool
THP_PyThreadState_HasStackSpace(PyThreadState *tstate, size_t size)
{
    CHECK(
        (tstate->datastack_top == NULL && tstate->datastack_limit == NULL)
        ||
        (tstate->datastack_top != NULL && tstate->datastack_limit != NULL)
    );
    return tstate->datastack_top != NULL &&
        size < (size_t)(tstate->datastack_limit - tstate->datastack_top);
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L2207
_PyInterpreterFrame *
THP_PyThreadState_BumpFramePointerSlow(PyThreadState *tstate, size_t size)
{
    if (THP_PyThreadState_HasStackSpace(tstate, size)) {
        _PyInterpreterFrame *res = (_PyInterpreterFrame *)tstate->datastack_top;
        tstate->datastack_top += size;
        return res;
    }
    if (size > INT_MAX/2) {
        PyErr_NoMemory();
        return NULL;
    }
    return (_PyInterpreterFrame *)push_chunk(tstate, (int)size);
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L2222
void
THP_PyThreadState_PopFrame(PyThreadState *tstate, _PyInterpreterFrame * frame)
{
    CHECK(tstate->datastack_chunk);
    PyObject **base = (PyObject **)frame;
    if (base == &tstate->datastack_chunk->data[0]) {
        _PyStackChunk *chunk = tstate->datastack_chunk;
        _PyStackChunk *previous = chunk->previous;
        // push_chunk ensures that the root chunk is never popped:
        CHECK(previous);
        tstate->datastack_top = &previous->data[previous->top];
        tstate->datastack_chunk = previous;
        THP_PyObject_VirtualFree(chunk, chunk->size);
        tstate->datastack_limit = (PyObject **)(((char *)previous) + previous->size);
    }
    else {
        CHECK(tstate->datastack_top);
        CHECK(tstate->datastack_top >= base);
        tstate->datastack_top = base;
    }
}


#endif

#if IS_PYTHON_3_11_PLUS

const uint8_t* THP_PyOpcode_Caches = _PyOpcode_Caches;
const int THP_PyOpcode_Caches_size = sizeof(_PyOpcode_Caches) / sizeof(uint8_t);

#else

const uint8_t* THP_PyOpcode_Caches = NULL;
const int THP_PyOpcode_Caches_size = 0;

#endif

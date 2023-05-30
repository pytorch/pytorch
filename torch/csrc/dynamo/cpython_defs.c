#include <torch/csrc/dynamo/cpython_defs.h>

#if IS_PYTHON_3_11_PLUS

#define Py_BUILD_CORE
#include <internal/pycore_pystate.h>
#define NEED_OPCODE_TABLES // To get _PyOpcode_Deopt
#include <internal/pycore_opcode.h>
#undef NEED_OPCODE_TABLES
#undef Py_BUILD_CORE

// As a simple way to reduce the impact of ABI changes on the CPython side, this check forces
// us to manually re-check that the function didn't change on the next major version
#if PY_VERSION_HEX >= 0x030C0000 // 3.12
#error "Please ensure that the functions below still match the CPython implementation for 3.12"
#endif

// https://github.com/python/cpython/blob/a7715ccfba5b86ab09f86ec56ac3755c93b46b48/Objects/frameobject.c#L1079
static int
_PyFrame_OpAlreadyRan(_PyInterpreterFrame *frame, int opcode, int oparg)
{
    // This only works when opcode is a non-quickened form:
    assert(_PyOpcode_Deopt[opcode] == opcode);
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

// https://github.com/python/cpython/blob/a7715ccfba5b86ab09f86ec56ac3755c93b46b48/Objects/frameobject.c#L1182
int
THP_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame) {
    /* Merge fast locals into f->f_locals */
    PyObject *locals;
    PyObject **fast;
    PyCodeObject *co;
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
                assert(value != NULL && PyCell_Check(value));
                value = PyCell_GET(value);
            }
            else if (kind & CO_FAST_CELL) {
                // Note that no *_DEREF ops can happen before MAKE_CELL
                // executes.  So there's no need to duplicate the work
                // that MAKE_CELL would otherwise do later, if it hasn't
                // run yet.
                if (value != NULL) {
                    if (PyCell_Check(value) &&
                            _PyFrame_OpAlreadyRan(frame, MAKE_CELL, i)) {
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
            assert(value == NULL);
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

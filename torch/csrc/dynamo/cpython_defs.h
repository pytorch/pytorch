#pragma once

#include <Python.h>
#include <torch/csrc/utils/python_compat.h>

// Functions that need to be copied from the CPython source
// should go in cpython_defs.c. Copying is required when, e.g.,
// we need to call internal CPython functions that are not exposed.

#if IS_PYTHON_3_11_PLUS

#include <internal/pycore_frame.h>

static int _PyFrame_OpAlreadyRan(
    _PyInterpreterFrame* frame,
    int opcode,
    int oparg);

int THP_PyFrame_FastToLocalsWithError(_PyInterpreterFrame* frame);

#endif

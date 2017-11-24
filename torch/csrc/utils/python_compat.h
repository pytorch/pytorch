#pragma once

#include <Python.h>

// https://bugsfiles.kde.org/attachment.cgi?id=61186
#if PY_VERSION_HEX >= 0x03020000
#define THPUtils_parseSlice(SLICE, LEN, START, STOP, LENGTH, STEP) \
  (PySlice_GetIndicesEx(SLICE, LEN, START, STOP, LENGTH, STEP) == 0)
#else
#define THPUtils_parseSlice(SLICE, LEN, START, STOP, LENGTH, STEP) \
  (PySlice_GetIndicesEx((PySliceObject*)SLICE, LEN, START, STOP, LENGTH, STEP) == 0)
#endif

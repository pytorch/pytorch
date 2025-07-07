#include <torch/csrc/utils/python_compat.h>
#if defined(_WIN32) && IS_PYTHON_3_11_PLUS
#define Py_BUILD_CORE
#define NEED_OPCODE_TABLES // To get _PyOpcode_Deopt, _PyOpcode_Caches

#if IS_PYTHON_3_13_PLUS
#include <cpython/code.h> // To get PyUnstable_Code_GetFirstFree
#define NEED_OPCODE_METADATA
#include "internal/pycore_opcode_metadata.h"
#undef NEED_OPCODE_METADATA
#else
#include "internal/pycore_opcode.h"
#endif

#undef NEED_OPCODE_TABLES
#undef Py_BUILD_CORE
#endif

#include <torch/csrc/utils/python_compat.h>
#if defined(_WIN32) && IS_PYTHON_3_11_PLUS
#define Py_BUILD_CORE
#define NEED_OPCODE_TABLES
#include "internal/pycore_opcode.h"
#endif

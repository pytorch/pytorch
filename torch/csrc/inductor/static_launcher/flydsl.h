#pragma once

#if defined(USE_ROCM)
#include <torch/csrc/python_headers.h>

bool FlyDSLMMFp16Bf16CWrapper_init(PyObject* module);
#endif

#pragma once

#if defined(USE_ROCM)
#include <torch/csrc/python_headers.h>

bool FlyDSLCWrapper_init(PyObject* module);
#endif

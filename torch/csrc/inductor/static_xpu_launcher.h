#pragma once
#if defined(USE_XPU)
#include <torch/csrc/python_headers.h>

bool StaticCudaLauncher_init(PyObject* module);
#endif

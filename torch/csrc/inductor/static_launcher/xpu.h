#pragma once
#if defined(USE_XPU)
#include <torch/csrc/python_headers.h>

bool StaticXpuLauncher_init(PyObject* module);
#endif

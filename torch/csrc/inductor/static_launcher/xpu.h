#pragma once
#if defined(USE_XPU)
#include <torch/csrc/inductor/cpp_wrapper/device_internal/xpu.h>
#include <torch/csrc/python_headers.h>

bool StaticXpuLauncher_init(PyObject* module);
#endif

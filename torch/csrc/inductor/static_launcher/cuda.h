#pragma once
#if defined(USE_CUDA)
#include <torch/csrc/inductor/cpp_wrapper/device_internal/cuda.h>
#include <torch/csrc/python_headers.h>

bool StaticCudaLauncher_init(PyObject* module);
#endif

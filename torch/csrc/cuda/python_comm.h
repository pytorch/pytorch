#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/CUDAContext.h>

#include <vector>

namespace torch { namespace cuda { namespace python {

void initCommMethods(PyObject *module);

std::vector<at::cuda::CUDAStream> py_object_to_cuda_streams(
    py::object py_streams);

}}}

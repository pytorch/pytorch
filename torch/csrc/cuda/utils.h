#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/python_numbers.h>

#include <vector>

std::vector<std::optional<at::cuda::CUDAStream>>
THPUtils_PySequence_to_CUDAStreamList(PyObject* obj);

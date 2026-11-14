#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::multiprocessing {

const PyMethodDef* python_functions();

} // namespace torch::multiprocessing

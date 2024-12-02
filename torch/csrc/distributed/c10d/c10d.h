#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::distributed::c10d {

PyMethodDef* python_functions();

} // namespace torch::distributed::c10d

#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::distributed::rpc::testing {

PyMethodDef* python_functions();

} // namespace torch::distributed::rpc::testing

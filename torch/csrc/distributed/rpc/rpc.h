#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {
namespace distributed {
namespace rpc {

PyMethodDef* python_functions();

} // namespace rpc
} // namespace distributed
} // namespace torch

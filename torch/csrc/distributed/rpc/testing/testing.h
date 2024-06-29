#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {
namespace distributed {
namespace rpc {
namespace testing {

PyMethodDef* python_functions();

} // namespace testing
} // namespace rpc
} // namespace distributed
} // namespace torch

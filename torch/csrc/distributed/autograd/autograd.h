#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {
namespace distributed {
namespace autograd {

constexpr auto kDistAutogradBackwardProfilingKey =
    "torch::distributed::autograd::backward";

PyMethodDef* python_functions();

} // namespace autograd
} // namespace distributed
} // namespace torch

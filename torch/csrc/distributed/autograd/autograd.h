#pragma once

#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/python_headers.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::variable_list;

constexpr auto kDistAutogradBackwardProfilingKey =
    "torch::distributed::autograd::backward";

TORCH_API void backward(
    int64_t context_id,
    const variable_list& roots,
    bool retain_graph = false);

PyMethodDef* python_functions();

} // namespace autograd
} // namespace distributed
} // namespace torch

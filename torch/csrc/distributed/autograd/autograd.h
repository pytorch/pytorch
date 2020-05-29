#pragma once

#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::variable_list;

TORCH_API void backward(
    int64_t context_id,
    const variable_list& roots,
    bool retain_graph = false);

} // namespace autograd
} // namespace distributed
} // namespace torch

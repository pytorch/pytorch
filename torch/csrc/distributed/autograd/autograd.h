#pragma once

#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::variable_list;

/// C++ API of Distributed Autograd that kicks off the distributed backward pass
/// using the provided roots. This currently implements the
/// :ref:`fast-mode-algorithm` which assumes all RPC messages sent in the same
/// distributed autograd context across workers would be part of the autograd
/// graph during the backward pass.
///
/// We use the provided roots to discover the autograd graph and compute
/// appropriate dependencies. This method blocks until the entire
/// autograd computation is done.
/// This function accumulates gradients in the leaves - you might need to zero
/// them before calling it.
///
/// \param context_id The autograd context id for which we should retrieve the
///                   gradients.
/// \param roots Tensors which represent the roots of the autograd computation.
///              All the tensors should be scalars.
/// \param retain_graph If `false`, the graph used to compute the grad will be
///                     freed. Note that in nearly all cases setting this
///                     option to `true` is not needed and often can be worked
///                     around in a much more efficient way. Usually, you need
///                     to set this to `true` to run backward multiple times.
TORCH_API void backward(
    int64_t context_id,
    const variable_list& roots,
    bool retain_graph = false);

} // namespace autograd
} // namespace distributed
} // namespace torch

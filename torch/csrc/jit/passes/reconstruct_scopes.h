/** \brief TBD
 *
 * TBD
 */
#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

TORCH_API void ReconstructScopes(
    script::Module& module,
    Graph& g,
    const std::string& prefix);

}
} // namespace torch

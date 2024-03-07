/** \brief This file defines freezing Torchscript module API.
 *
 * This API has python-binding and can be invoked directly or as a part of
 * general optimization pipeline.
 */
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

/** \brief Freeze Module, i.e., Assume all attributes are constants.
 *
 * Freezing module is a functionality that allows the JIT to internalize
 * immutable attributes. Combined with inlining, the module is aggressively
 * optimized and significant overhead is optimized away. The freezeModule API
 * produces a cloned frozen module.
 */

namespace torch {
namespace jit {

TORCH_API Module freeze_module(
    const Module& module,
    std::vector<std::string> preservedAttrs = std::vector<std::string>(),
    bool freezeInterfaces = true,
    bool preserveParameters = false);

// Clone-free version of freeze_module. This modifies the module inplace.
// Use this version to avoid extra memory usage incurred by cloning the module.
TORCH_API void freeze_module_inplace(
    Module* module,
    std::vector<std::string> preservedAttrs = std::vector<std::string>(),
    bool freezeInterfaces = true,
    bool preserveParameters = false);
} // namespace jit
} // namespace torch

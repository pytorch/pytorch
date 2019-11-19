/** \brief This file defines freezing Torchscript module API.
 *
 * This API has python-binding and can be invoked directly or as a part of
 * general optimization pipeline.
 */
#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

/** \brief Freeze Module, i.e., Assume all atrributes are constants.
 *
 * Freezing module is a functionality that allows the JIT to internalize
 * imutable attributes. Combined with inlinig, the module is aggressively
 * optimized and significant overhead is optimized away. The freezeModule API
 * produces a cloned frozen module.
 */

namespace torch {
namespace jit {

TORCH_API script::Module freeze_module(const script::Module& module);

} // namespace jit
} // namespace torch

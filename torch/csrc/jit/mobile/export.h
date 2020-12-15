#pragma once

#include <torch/csrc/jit/mobile/module.h>
#include <set>
#include <string>

namespace torch::jit::mobile {

/**
 * Given a torch::jit::mobile::Module, return a set of operator names
 * (with overload name) that are used by any method in this mobile
 * Mobile. This method runs through the bytecode for all methods
 * in the specified model (module), and extracts all the root
 * operator names. Root operators are operators that are called
 * directly by the model (as opposed to non-root operators, which
 * may be called transitively by the root operators).
 *
 */
TORCH_API std::set<std::string> _export_operator_list(
    torch::jit::mobile::Module& module);

} // namespace torch::jit::mobile

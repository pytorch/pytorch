/** \brief A pass to reconstruct scopes of nodes from their inline callstacks.
 *
 * The pass takes the root module and a graph and for every graph node with
 * non-empty inline call-stack it computes the scope from this callstack.
 *
 * Callstack can be thought of as a stack of pointers to Function, and Function
 * in a general case may not be a part of any module. That's why this pass
 * requires a root module to be passed in - we can traverse all methods of the
 * module and its submodules and then recognize these methods in callstacks.
 *
 * Scope can be thought of as a stack of strings, so we basically converting a
 * pointer to Function to a string, or in other words trying to find a name for
 * a function in this module hierarchy.
 *
 * The produced scopes look like:
 * top.submod1.function1/top.submod1.subsubmod1.function2
 *
 * 'top' is the name we use for the root module itself, and it can be customized
 * with an optional third argument of the pass.
 *
 * The pass would not change anything if inlining has not been run on the graph.
 */
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void ReconstructScopes(
    const Module& module,
    Graph& g,
    const std::string& prefix);

} // namespace jit
} // namespace torch

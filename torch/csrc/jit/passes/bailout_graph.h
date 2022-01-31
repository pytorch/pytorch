#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

#include <list>
#include <vector>

namespace torch {
namespace jit {

// Replaces prim::Guard nodes with prim::BailOut nodes and
// computes sets of inputs needed to resume execution at
// bailout points
TORCH_API void InsertBailOuts(std::shared_ptr<Graph> graph);

// Builds a bailout graph into `target` (which is an empty graph)
// for a given bailout point `bailout_index`
// from the original graph `orig` (the original unoptimized graph)
// BailOut graphs allow Interpreter to resume
// execution of the (un/de)optimized graph (i.e.
// a graph that doesn't rely on any assumptions derived from
// on profiling information) from a given BailOut point
// should any of the assumptions fail for an actual input.
TORCH_API std::shared_ptr<Graph> BuildBailOutGraphFrom(
    int64_t bailout_index,
    const std::shared_ptr<Graph>& orig,
    const std::shared_ptr<Graph>& target);
} // namespace jit
} // namespace torch

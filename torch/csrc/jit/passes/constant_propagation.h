#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// Runs constant propagation on all objects unless ignore_custom_classes is
// specified as true, in which case user defined classes are skipped.  This is
// useful to prevent early fusion of packing operations, which end up lowering
// away information about their constructors (e.g. packed::linear_clamp_prepack
// and prepacked::conv2d_clamp_prepack)
// Returns True if the pass made a change to the graph
TORCH_API bool ConstantPropagation(
    std::shared_ptr<Graph>& graph,
    bool ignore_custom_classes = false);

// runs constant propagation only on ops that have non-aliasing inputs & outputs
// Returns True if the pass made a change to the graph
TORCH_API bool ConstantPropagationImmutableTypes(std::shared_ptr<Graph>& graph);

// Runs the node if its inputs are constants. Callers of this function must
// make their own determination if constant prop is appropriate - for example
// non-deterministic ops or ops with side effects.  If ignore_custom_classes is
// specified, nodes that output user defined classes are not run.
TORCH_API std::optional<Stack> runNodeIfInputsAreConstant(
    const Node* node,
    bool ignore_custom_classes = false,
    AliasDb* db = nullptr);

} // namespace torch::jit

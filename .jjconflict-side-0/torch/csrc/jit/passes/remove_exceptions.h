#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// Considering prim::RaiseException nodes unreachable, simplify prim::If nodes
// when one of the branches contains prim::RaiseException.
//
// This pass is illegal in general case as the modified graph might not throw
// an exception that the original graph would throw. The purpose of the pass is
// to cleanup the graph in a "risky" way by removing pathways leading to
// RaiseExceptions nodes. In some sense, this pass could be considered as a
// "Release" mode, while the original graph was in a "Debug" mode.
// The pass should only be used when such transformation is guaranteed to be
// safe by some other mechanisms. For instance, when we know exact shapes of
// tensors flowing through the graph and tensors with such shapes never cause
// exceptions.
TORCH_API void EliminateExceptions(std::shared_ptr<Graph>& graph);

} // namespace torch::jit

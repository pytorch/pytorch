#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Peephole Optimizes Dict Ops such as len() and __getitem__
// 1. getitem optimizations
// Given a function like this:
//     def foo():
//         d = {0 : 1}
//         x = d[0]
//         return x
// This pass produces (after dead code elimination):
//     def foo(a, b):
//         return 1
//
// This optimization can only happen if the dict is not modified
// and the dict has constant, non overlapping keys.
//
// 2. len optimizations
// Given a function like this:
//     def foo():
//         d = {0 : 1}
//         return len(d)
// This pass produces (after dead code elimination):
//     def foo():
//         return 1
//
// This has the same requirements as the getitem optimizations.
//
// Currently this is invoked as part of PeepholeOptimize
// return true if graph is modified.
TORCH_API bool PeepholeOptimizeDictIdioms(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch

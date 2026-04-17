#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// Peephole Optimizes List ops such as len(li) and li[1].
// 1. Construct/Unpack optimizations
// Given a function like this:
//    def foo(a, b):
//        li = [a, b]
//        x, y = li
//        return x, y
// This pass produces (after dead code elimination):
//    def foo(a, b):
//        return a, b
//
// This is only applied to lists that are not modified.
//
// 2. getitem optimizations
// Given a function like this:
//     def foo(a, b):
//         li = [a, b]
//         x = li[0]
//         return x
// This pass produces (after dead code elimination):
//     def foo(a, b):
//         return a
//
// This optimization can only happen if the list is not modified.
//
// 3. len optimizations
// Given a function like this:
//     def foo():
//         li = [1, 2]
//         return len(li)
// This pass produces (after dead code elimination):
//     def foo():
//         return 2
//
// This has the same requirements as the getitem optimizations.
//
// 4. ListConstruct + ListConstruct
// Given a function like this:
//     def foo():
//         return [1, 2] + [3, 4]
// This pass produces (after dead code elimination):
//     def foo():
//         return [1, 2, 3, 4]
//
// This is only applied to lists that are not modified.
//
// 5. Slice
// Given a function like this:
//     def foo():
//         return [1, 2, 3, 4, 5][0:2]
// This pass produces (after deadcode elimination):
//     def foo():
//         return [1, 2]
//
// Currently this is invoked as part of PeepholeOptimize
// return true if graph is modified.
// If `refine_list_len` is true will attempt to refine the len of lists through
// len comparisons and assertions. This does not generally optimize pytorch
// programs so it is not called by default in PeepholeOptimize.
TORCH_API bool PeepholeOptimizeListIdioms(
    const std::shared_ptr<Graph>& graph,
    bool refine_list_len = false);

} // namespace torch::jit

#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/tree_views.h>

namespace torch {
namespace jit {
namespace script {

// This is an AST-to-AST transform that ensures that all return statements
// are at the end of the natural control-flow of the program.
//
// Since the return is at the end of the function, it is equivalent
// to simply assigning the returned value to to a special `$return` variable
// that is universally set to be the output of the function.
//
// This transform is only intended to support a subset of control-flow
// structures to make the transformation both easy to do _and_ easy to
// explain to users of TorchScript. The second constraint is important: if
// it is unclear what is allowed users will get the impression that the
// subset is difficult to use.
//
//   if <cond>:
//     <true>
//   else:
//     <false>
//   <rest>
//
// In particular we allow:
// 1. If statements where neither <true> nor <false> branch returns.
// 2. If statements where both <true> and <false> always return.
// 3. An 'early return' if statement where <true> always returns <false> is
// empty, and <rest> always returns.
//
// We do not allow returns from loops in any case.
//
// This pass handles the following cases as follows:
//
// 1. Neither branch returns so we can just leave the branches as is
// 2. Both branches return, so we recursively transform the program such that
// <true> and <false>'s final action is to return. We then delete <rest>
// because the code is dead. The remaining program preserves the inductive
// property that its last action is to return since both branches end in a
// return.
// 3. In this case we know that <true> and <rest> always returns, and <false> is
// empty.
//    We transform the graph to:
//    if <cond>:
//       <true>
//     else:
//       <rest>
//    Now it is another instance of case (2).

TORCH_API List<Stmt> moveAllReturnsToEnd(const List<Stmt>& stmts);

} // namespace script
} // namespace jit
} // namespace torch

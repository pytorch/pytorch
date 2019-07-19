#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/final_returns.h>

namespace torch {
namespace jit {
namespace script {

struct ReturnInfo {
  bool returns_; // true - all paths through stmts_ always return
                 // false - all paths through stmts_ do not return
  List<Stmt> stmts_;
};

void checkNoReturn(const TreeRef& ref) {
  if (ref->kind() == TK_RETURN) {
    throw ErrorReport(ref) << "return is not allowed from a loop";
  }
  // do not search into first-class functions
  if (ref->kind() == TK_DEF) {
    return;
  }

  for (const TreeRef& child : ref->trees()) {
    checkNoReturn(child);
  }
}

// transform stmts so that its last action is to return or report that it
// never returns.
// return_none - if true, add an implicit `return None` to the end of the block
//   this handles the case where the return is implicit at the end of the
//   function.
ReturnInfo makeReturnsFinal(
    const SourceRange& range,
    at::ArrayRef<TreeRef> stmts,
    bool return_none);
ReturnInfo makeReturnsFinal(const List<Stmt>& stmts, bool return_none) {
  return makeReturnsFinal(stmts.range(), stmts.get()->trees(), return_none);
}
ReturnInfo makeReturnsFinal(
    const SourceRange& range,
    at::ArrayRef<TreeRef> stmts,
    bool return_none) {
  at::SmallVector<TreeRef, 4> changed;
  changed.reserve(stmts.size());
  for (size_t i = 0; i < stmts.size(); ++i) {
    const TreeRef& stmt = stmts[i];
    switch (stmt->kind()) {
      case TK_IF: {
        auto if_stmt = If(stmt);
        auto true_final = makeReturnsFinal(if_stmt.trueBranch(), false);
        // (3) early return an if statement without an else block:
        if (true_final.returns_ && if_stmt.falseBranch().size() == 0) {
          auto rest_final =
              makeReturnsFinal(range, stmts.slice(i + 1), return_none);
          if (!rest_final.returns_) {
            throw ErrorReport(if_stmt)
                << "This if statement performs an early return, but the block of code that follows it does not return."
                << " Early returns are only allowed when the block following them also returns";
          }
          changed.emplace_back(
              if_stmt.withNewBranches(true_final.stmts_, rest_final.stmts_));
          return {true, List<Stmt>::unsafeCreate(range, std::move(changed))};
        }

        auto false_final = makeReturnsFinal(if_stmt.falseBranch(), false);
        // (1) neither branch returns just keep processing the block
        if (!true_final.returns_ && !false_final.returns_) {
          changed.emplace_back(if_stmt);
          break;
        }
        // (2) all branches return
        if (true_final.returns_ && false_final.returns_) {
          changed.emplace_back(
              if_stmt.withNewBranches(true_final.stmts_, false_final.stmts_));
          return {true, List<Stmt>::unsafeCreate(range, std::move(changed))};
        }
        throw ErrorReport(if_stmt)
            << "This if statement contains some paths that return and some paths that do not. "
            << "If statements must either entirely return or never return";
      } break;
      case TK_WHILE:
      case TK_FOR:
        changed.emplace_back(stmt);
        checkNoReturn(stmt);
        break;
      case TK_RETURN:
        changed.emplace_back(stmt);
        // ignore the rest the the block, it is dead.
        return {true, List<Stmt>::unsafeCreate(range, std::move(changed))};
      default:
        changed.emplace_back(stmt);
        break;
    }
  }
  if (return_none) {
    // add an implicit return none node
    changed.emplace_back(
        Return::create(range, Expr(Compound::create(TK_NONE, range, {}))));
  }
  // we reach the end of the block, no returns have happened
  // unless we just inserted a return_none implicit return.
  return {return_none, List<Stmt>::unsafeCreate(range, std::move(changed))};
}

List<Stmt> moveAllReturnsToEnd(const List<Stmt>& stmts) {
  return makeReturnsFinal(stmts, true).stmts_;
}

} // namespace script
} // namespace jit
} // namespace torch

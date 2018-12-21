#include <torch/csrc/jit/script/final_returns.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace script {

struct ReturnInfo {
  bool returns_;
  List<Stmt> stmts_;
};

void checkNoReturn(const TreeRef& ref) {
  if (ref->kind() == TK_RETURN)
    throw ErrorReport(ref) << "return is not allowed from a loop.";
  for(const TreeRef& child : ref->trees()) {
    checkNoReturn(child);
  }
}

void failReturns(const If& if_stmt, const char * what) {
  throw ErrorReport(if_stmt)
        << what << " contains some paths that return and some paths that do not. "
        << "If statements must either entirely return or never return.";
}


ReturnInfo makeReturnsFinal(const SourceRange& range, at::ArrayRef<TreeRef> stmts, bool return_none);
ReturnInfo makeReturnsFinal(const List<Stmt>& stmts, bool return_none) {
  return makeReturnsFinal(stmts.range(), stmts.get()->trees(), return_none);
}
ReturnInfo makeReturnsFinal(const SourceRange& range, at::ArrayRef<TreeRef> stmts, bool return_none) {
  std::vector<TreeRef> changed;
  changed.reserve(stmts.size());
  for(size_t i = 0; i < stmts.size(); ++i) {
    const TreeRef& stmt = stmts[i];
    switch(stmt->kind()) {
      case TK_IF: {
        auto if_stmt = If(stmt);
        auto true_final = makeReturnsFinal(if_stmt.trueBranch(), false);
        // early return an if statement without an else block:
        if (true_final.returns_ && if_stmt.falseBranch().size() == 0) {
          auto rest_final = makeReturnsFinal(range, stmts.slice(i + 1), return_none);
          if (!rest_final.returns_) {
            failReturns(if_stmt, "The enclosing if statement");
          }
          changed.emplace_back(if_stmt.withNewBranches(true_final.stmts_, rest_final.stmts_));
          return {true, List<Stmt>::unsafeCreate(range, std::move(changed))};
        }

        auto false_final = makeReturnsFinal(if_stmt.falseBranch(), false);
        if (!true_final.returns_ && !false_final.returns_) {
          changed.emplace_back(if_stmt);
          break;
        }
        if (true_final.returns_ && false_final.returns_) {
          changed.emplace_back(if_stmt.withNewBranches(true_final.stmts_, false_final.stmts_));
          return {true, List<Stmt>::unsafeCreate(range, std::move(changed))};
        }
        failReturns(if_stmt, "This if statement");
      } break;
      case TK_WHILE:
        changed.emplace_back(stmt);
        checkNoReturn(stmt);
        break;
      case TK_RETURN:
        changed.emplace_back(stmt);
        // ignore the rest the the block, all paths return
        return {true, List<Stmt>::unsafeCreate(range, std::move(changed))};
      default:
        changed.emplace_back(stmt);
        break;
    }
  }
  if (return_none) {
    // add an implicit return none node
    changed.emplace_back(Return::create(range, Expr(Compound::create(TK_NONE, range, {}))));
  }
  // we reach the end of the block, no returns have happened
  return {return_none, List<Stmt>::unsafeCreate(range, std::move(changed))};
}

List<Stmt> moveAllReturnsToEnd(const List<Stmt>& stmts) {
  return makeReturnsFinal(stmts, true).stmts_;
}

} // namespace script
} // namespace jit
} // namespace torch

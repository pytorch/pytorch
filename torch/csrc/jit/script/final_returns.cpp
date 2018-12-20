#include <torch/csrc/jit/script/final_returns.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace script {

struct ReturnsInfo {
  enum Status {
    AllReturn, OneNoReturn, NoReturn
  };
  ReturnsInfo(Status status, List<Stmt> stmts, std::vector<bool> path = {})
  : status_(status), stmts_(std::move(stmts)), path_(std::move(path)) {}
  ReturnsInfo(Status status, const SourceRange& range, std::vector<TreeRef> trees, std::vector<bool> path = {})
  : ReturnsInfo(status, List<Stmt>::unsafeCreate(range, std::move(trees)), std::move(path)) {}

  Status status_;
  List<Stmt> stmts_;
  // if OneNoReturn, the path through the final if statements to find
  // the block that does not return.
  std::vector<bool> path_;
};

void checkNoReturn(const TreeRef& ref) {
  if (ref->kind() == TK_RETURN)
    throw ErrorReport(ref) << "return is not allowed from a loop.";
  for(const TreeRef& child : ref->trees()) {
    checkNoReturn(child);
  }
}

TreeRef spliceToEnd(const std::vector<bool>& path, size_t idx, const If& if_stmt, const List<Stmt>& rest);
List<Stmt> spliceToEnd(const std::vector<bool>& path, size_t idx, const List<Stmt>& branch, const List<Stmt>& rest) {
  std::vector<TreeRef> stmts = branch.get()->trees();
  if (idx == path.size()) {
    stmts.insert(stmts.end(), rest.get()->trees().begin(), rest.get()->trees().end());
  } else {
    auto last = If(stmts.back());
    stmts.back() = spliceToEnd(path, idx, last, rest);
  }
  return List<Stmt>::unsafeCreate(branch.range(), std::move(stmts));
}
TreeRef spliceToEnd(const std::vector<bool>& path, size_t idx, const If& if_stmt, const List<Stmt>& rest) {
  if (path[idx]) {
    auto branch = spliceToEnd(path, idx + 1, if_stmt.trueBranch(), rest);
    return if_stmt.withNewBranches(branch, if_stmt.falseBranch());
  } else {
    auto branch = spliceToEnd(path, idx + 1, if_stmt.falseBranch(), rest);
    return if_stmt.withNewBranches(if_stmt.trueBranch(), branch);
  }
}

void checkAlwaysReturns(const If& if_stmt, ReturnsInfo::Status status) {
  if (status != ReturnsInfo::AllReturn) {
    throw ErrorReport(if_stmt)
        << "when returning from a (possibly nested) if statement there must be at most one branch that does not return. Here there are multiple branches that do not return";
  }
}

ReturnsInfo makeReturnsFinal(const List<Stmt>& stmts) {
  std::vector<TreeRef> changed;
  changed.reserve(stmts.size());
  for(size_t i = 0; i < stmts.size(); ++i) {
    const TreeRef& stmt = stmts[i];
    switch(stmt->kind()) {
      case TK_IF: {
        auto if_stmt = If(stmt);
        auto true_final = makeReturnsFinal(if_stmt.trueBranch());
        auto false_final = makeReturnsFinal(if_stmt.falseBranch());
        if (true_final.status_ == ReturnsInfo::AllReturn &&
            false_final.status_ == ReturnsInfo::AllReturn) {
          changed.emplace_back(if_stmt.withNewBranches(true_final.stmts_, false_final.stmts_));
          return ReturnsInfo(ReturnsInfo::AllReturn, stmts.range(), std::move(changed));
        } else if (true_final.status_ == ReturnsInfo::NoReturn &&
            false_final.status_ == ReturnsInfo::NoReturn) {
          changed.emplace_back(stmt);
        } else {
          std::vector<bool> path;
          if (false_final.status_ != ReturnsInfo::AllReturn) {
            checkAlwaysReturns(if_stmt, true_final.status_);
            path.push_back(false);
            path.insert(path.end(), false_final.path_.begin(), false_final.path_.end());
          } else {
            AT_CHECK(true_final.status_ != ReturnsInfo::AllReturn);
            checkAlwaysReturns(if_stmt, false_final.status_);
            path.push_back(true);
            path.insert(path.end(), true_final.path_.begin(), true_final.path_.end());
          }
          at::ArrayRef<TreeRef> stmts_view = stmts.get()->trees();
          auto rest_final = makeReturnsFinal(List<Stmt>::unsafeCreate(
              stmts.range(), stmts_view.slice(i + 1).vec()));
          path.insert(
              path.end(), rest_final.path_.begin(), rest_final.path_.end());
          auto new_if =
              if_stmt.withNewBranches(true_final.stmts_, false_final.stmts_);
          TreeRef spliced = spliceToEnd(path, 0, new_if, rest_final.stmts_);
          changed.emplace_back(spliced);
          if (rest_final.status_ == ReturnsInfo::AllReturn) {
            return ReturnsInfo(ReturnsInfo::AllReturn, stmts.range(), std::move(changed));
          } else {
            path.insert(path.end(), rest_final.path_.begin(), rest_final.path_.end());
            return ReturnsInfo(ReturnsInfo::OneNoReturn, stmts.range(), std::move(changed), std::move(path));
          }
        }
      } break;
      case TK_WHILE:
        changed.emplace_back(stmt);
        checkNoReturn(stmt);
        break;
      case TK_RETURN:
        changed.emplace_back(stmt);
        // ignore the rest the the block, all paths return
        return ReturnsInfo(ReturnsInfo::AllReturn, stmts.range(), std::move(changed));
      default:
        changed.emplace_back(stmt);
        break;
    }
  }
  // we reach the end of the block, no returns have happened, so NoReturn
  return ReturnsInfo(ReturnsInfo::NoReturn, stmts.range(), std::move(changed));
}

List<Stmt> moveAllReturnsToEnd(const List<Stmt>& stmts) {
  return makeReturnsFinal(stmts).stmts_;
}

} // namespace script
} // namespace jit
} // namespace torch

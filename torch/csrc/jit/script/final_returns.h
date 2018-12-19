#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/tree_views.h>

namespace torch {
namespace jit {
namespace script {

List<Stmt> moveAllReturnsToEnd(const List<Stmt>& stmts);

}
} // namespace jit
} // namespace torch

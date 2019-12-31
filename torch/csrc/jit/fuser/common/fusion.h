#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

// TODO: make destructor free stmts
// TODO: add regions (lists of exprs)
// TODO: functions for inserting expressions at, moving them
// TODO: replaceAllUsesWith for values
// TODO: DCE pass
// TODO: register exprs and vals with the fusion (implicitly)
struct TORCH_API Fusion {
  Fusion() = default;

  // Not copyable
  Fusion(const Fusion& other) = delete;
  Fusion& operator=(const Fusion& other) = delete;

  Fusion(Fusion&& other) = default;
  Fusion& operator=(Fusion&& other) = default;

  ~Fusion() = default;

  void appendExpr(Expr* expr) { expr_list_.push_back(expr); }
  std::vector<Expr*> list() { return expr_list_; }

private:
  std::unordered_map<StmtNameType, Val*> val_map_;
  std::unordered_map<StmtNameType, Expr*> expr_map_;

  StmtNameType val_name_counter_ = 0;
  StmtNameType expr_name_counter_ = 0;

  StmtNameType getValName() { return val_name_counter_++; }
  StmtNameType getExprName() { return expr_name_counter_++; }

  std::vector<Expr*> expr_list_;
};

// struct TORCH_API Region { // list of nodes (expressions)
// };


//   // fusion: removeNode, addNodeBefore, addNodeAfter, addNodeAt
// };

}}} // torch::jit::fuser

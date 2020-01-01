#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/util/Exception.h>

#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// TODO: add regions (lists of exprs)
// TODO: functions for inserting expressions at, moving them
// TODO: replaceAllUsesWith for values
// TODO: DCE pass
struct TORCH_API Fusion {
  Fusion() {
    region_ = new Region{};
    region_->setFusion(this);
  }

  // Not copyable
  Fusion(const Fusion& other) = delete;
  Fusion& operator=(const Fusion& other) = delete;

  Fusion(Fusion&& other) = default;
  Fusion& operator=(Fusion&& other) = default;

  ~Fusion(){
    delete region_;

    for (auto it = val_map_.begin(); it != val_map_.end(); ++it) {
      delete it->first;
    }

    for (auto it = expr_map_.begin(); it != expr_map_.end(); ++it) {
      auto* expr = it->first;
      for (auto* region : expr->regions()) {
        delete region;
      }
      delete expr;
    }
  };

  void insertAtStart(Expr* expr) { region_->insertAtStart(expr); }
  void insertAtEnd(Expr* expr) { region_->insertAtEnd(expr) };
  void insertLeftBeforeRight(Expr* left, Expr* right) {
    region_->insertLeftBeforeRight(left, right);
  }
  void insertLeftAfterRight(Expr* left, Expr* right) {
    region_->insertLeftAfterRight(left, right);
  }

  StmtNameType registerStatement(Statement* stmt) {
    if (stmt->isVal()) {
      return registerVal(static_cast<Val*>(stmt));
    }

    return registerExpr(static_cast<Expr*>(stmt));
  }

  StmtNameType registerVal(Val* val) {
    if (val->fusion()) {
      TORCH_CHECK(inFusion(val));
      return val->name();
    }

    val->setFusion(this);
    val->setName(getValName());
    val_map_[val] = val->name();

    return val->name();
  }

  void registerRegion(Region* region) {
    if (region->fusion()) {
      TORCH_CHECK(region->fusion() == this);
      return;
    }

    region->setFusion(this);

    for (auto* input : region->inputs()) {
      registerStatement(input);
    }

    for (auto* expr : region->exprs()) {
      registerStatement(expr);
    }

    for (auto* output : region->outputs()) {
      registerStatement(output);
    }
  }

  StmtNameType registerExpr(Expr* expr){
    if (expr->fusion()) {
      TORCH_CHECK(inFusion(expr));
      return expr->name();
    }

    expr->setFusion(this);
    expr->setName(getExprName());
    expr_map_[expr] = expr->name();

    for (auto* input : expr->inputs()) {
      registerStatement(input);
    }

    for (auto* region : expr->regions()) {
      registerRegion(region);
    }

    for (auto* output : expr->outputs()) {
      registerStatement(output);
    }

    return expr->name();
  }

  bool inFusion(const Statement* stmt){
    return (stmt->fusion() == this);
  }

private:
  Region* region_ = nullptr;

  std::unordered_map<Val*, StmtNameType> val_map_;
  std::unordered_map<Expr*, StmtNameType> expr_map_;

  StmtNameType val_name_counter_ = 0;
  StmtNameType expr_name_counter_ = 0;

  StmtNameType getValName() { return val_name_counter_++; }
  StmtNameType getExprName() { return expr_name_counter_++; }
};

}}} // torch::jit::fuser

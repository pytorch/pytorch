#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/util/Exception.h>

#include <unordered_map>
#include <vector>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

// TODO: replaceAllUsesOfLeftWithRight(Val, Val) for values
// TODO: hasUses(Val)
// TODO: val->uses mapping? expr->location mapping?
// TODO: plumb through helpers, ex. Val.replaceAllUsesWith(Val)
//          (registration's make this easy)
// TODO: DCE pass
// TODO: printFusion (allow mem location inlining)
// TODO:
// TODO: comment
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

  // Functions for inserting expressions
  void insertAtStart(Expr* expr) { region_->insertAtStart(expr); }
  void insertAtEnd(Expr* expr) { region_->insertAtEnd(expr); }
  void insertLeftBeforeRight(Expr* left, Expr* right) {
    region_->insertLeftBeforeRight(left, right);
  }
  void insertLeftAfterRight(Expr* left, Expr* right) {
    region_->insertLeftAfterRight(left, right);
  }

  // Functions for adding inputs and outputs
  void addInput(Statement* input) { region_->addInput(input); }
  void addOutput(Statement* output) { region_->addOutput(output); }

  // Functions for querying / enumerating IR objets
  bool inFusion(const Statement* stmt){
    return (stmt->fusion() == this);
  }
  std::deque<Statement*>& inputs() noexcept { return region_->inputs(); }
  std::deque<Statement*>& outputs() noexcept { return region_->outputs(); }

  std::deque<Expr*>& exprs() noexcept { return region_->exprs(); }

  void print(std::ostream& os) {
    os << "Fusion{Inputs(" << std::endl;
    for (auto* input : inputs()) {
      os << input << std::endl;
    }

    os << ")->Body(" << std::endl;

    for (auto* expr : exprs()) {
      os << expr << std::endl;
    }

    // TODO: print outputs

  }


  // Functions for registering IR objects
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

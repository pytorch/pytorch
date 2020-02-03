#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/util/Exception.h>

#include <unordered_map>
#include <vector>
#include <stack>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

// TODO: replaceAllUsesOfLeftWithRight(Val, Val) for values
// TODO: hasUses(Val)
// TODO: val->uses mapping? expr->location mapping?
// TODO: plumb through helpers, ex. Val.replaceAllUsesWith(Val)
//          (registration makes this easy)
// TODO: DCE pass
// TODO: comment

struct Fusion;

struct TORCH_API FusionGuard{

public:
  static thread_local Fusion* cur_fusion;
  Fusion* prev_fusion;

  FusionGuard(Fusion* const fusion) {
    prev_fusion = cur_fusion;
    cur_fusion = fusion;
  }

  ~FusionGuard() {
    cur_fusion = prev_fusion;
  }

  static Fusion* getCurFusion() {
    return cur_fusion;
  }

};

struct TORCH_API Fusion : public IRInputOutput{
  Fusion() {}

  // Not copyable
  Fusion(const Fusion& other) = delete;
  Fusion& operator=(const Fusion& other) = delete;

  Fusion(Fusion&& other) = default;
  Fusion& operator=(Fusion&& other) = default;

  ~Fusion(){

    for (auto it = val_map_.begin(); it != val_map_.end(); ++it) {
      delete it->first;
    }

    for (auto it = expr_map_.begin(); it != expr_map_.end(); ++it) {
      delete it->first;
    }
  };

  //Probably this should be implicit in register Expr
  //i.e. if an expr is registered with an output
  //we should check if it's an output of any other expr.
  //If it is, we should delete that expr.
  void remove_expr(const Expr* expr){
    if(!inFusion(expr))
      throw std::runtime_error("Cannot remove expr as it is not registered with this fusion.");
    expr_map_.erase(expr);
    delete expr;
  }

  // Functions for adding inputs and outputs
  void addInput(const Val* input) {
    if(!inFusion(input))
      throw std::runtime_error("Cannot register input as it does not belong to this fusion.");
    IRInputOutput::addInput(input);
  }

  void addOutput(const Val* output) {
    if(!inFusion(output))
      throw std::runtime_error("Cannot register output as it does not belong to this fusion.");
    IRInputOutput::addOutput(output);
  }

  // Functions for querying / enumerating IR objets
  bool inFusion(const Statement* stmt){
    bool infusion = stmt->fusion() == this;

    if(stmt->isExpr())
      infusion &= expr_map_.find(static_cast<const Expr*>(stmt)) != expr_map_.end();
    if(stmt->isVal())
      infusion &= val_map_.find(static_cast<const Val*>(stmt)) != val_map_.end();

    return infusion;
  }

  StmtNameType registerStatement(const Statement* stmt) {
    if(inFusion(stmt))
      return stmt->name();

    if (stmt->isVal()) {
      return registerVal(static_cast<const Val*>(stmt));
    }else if(stmt->isExpr()){
      return registerExpr(static_cast<const Expr*>(stmt));
    }

    throw std::runtime_error("Could not register statement.");
    return UNINITIALIZED_STMTNAMETYPE;
  }

  //TODO: Lets topologically sort these.
  std::vector<const Expr*> exprs() const {
    std::vector<const Expr*> exprs;
    for(const auto it : expr_map_)
      exprs.push_back(it.first);
    return exprs;
  }

  // TODO: Simplify this function, we register values as soon as they're created
  StmtNameType registerVal(const Val* val) {
    if (val->fusion()) {
      TORCH_CHECK(inFusion(val)); //Registered with another fusion
      return val->name();
    }
    val_map_[val] = val->name();
    return val->name();
  }

  StmtNameType registerExpr(const Expr* expr){
    if (expr->fusion()) {
      TORCH_CHECK(inFusion(expr));
      return expr->name();
    }

    expr_map_[expr] = expr->name();

    for (const Val* input : expr->inputs()) {
      registerVal(input);
    }

    for (const Val* output : expr->outputs()) {
      registerVal(output);
    }

    return expr->name();
  }

private:
  std::unordered_map<const Val*, StmtNameType> val_map_;
  std::unordered_map<const Expr*, StmtNameType> expr_map_;

  StmtNameType val_name_counter_ = 0;
  StmtNameType expr_name_counter_ = 0;

  StmtNameType getValName() { return val_name_counter_++; }
  StmtNameType getExprName() { return expr_name_counter_++; }
};

  TORCH_API std::ostream& operator<<(std::ostream& os, const Fusion& fusion);
  TORCH_API std::ostream& operator<<(std::ostream& os, const std::deque<const Val*>& vals);

}}} // torch::jit::fuser

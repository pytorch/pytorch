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
  friend class Statement;
  friend class Val;
  friend class Expr;
  Fusion() = default;

  // Not copyable
  Fusion(const Fusion& other) = delete;
  Fusion& operator=(const Fusion& other) = delete;

  Fusion(Fusion&& other) = default;
  Fusion& operator=(Fusion&& other) = default;

  ~Fusion(){
    for(auto it = val_map_.begin(); it != val_map_.end(); ++it)
      delete it->first;
    for(auto it = expr_map_.begin(); it != expr_map_.end(); ++it)
      delete it->first;
  };

protected:
  StmtNameType addVal(const Val* val){
    auto it = val_map_.find(val);
    if( it != val_map_.end())
      return it->second;
    auto valname = getValName();
    val_map_.emplace(val, valname);
    return valname;
  }

  bool inFusion(const Statement* stmt){
    if(stmt->isVal()){
        if(val_map_.find(static_cast<const Val*>(stmt)) == val_map_.end())
          return false;
        return true;
      }else if(stmt->isExpr()){
        if(expr_map_.find(static_cast<const Expr*>(stmt)) == expr_map_.end()) 
          return false;
        return true;
      }
      throw std::runtime_error("Statement type not recognized.");
      
      
  }

  StmtNameType addExpr(const Expr* expr){
    auto it = expr_map_.find(expr);
    if(it != expr_map_.end())
      return it->second;

    
    for(std::vector<Statement*>::size_type i=0; i<expr->n_inputs(); ++i){
      const auto* stmt = expr->getInput(i);
      if(!inFusion(stmt))
        //TODO
        std::runtime_error(/* stmt.stringify() "+ */" found in expr but not found in fusion.");
    }
    
    for(std::vector<Statement*>::size_type i=0; i<expr->n_outputs(); ++i){
      const auto* stmt = expr->getOutput(i);
      if(!inFusion(stmt))
        //TODO
        std::runtime_error(/* stmt.stringify() "+ */" found in expr but not found in fusion.");
    }

    auto exprname = getExprName();
    expr_map_.emplace(expr, exprname);
    return exprname;
  }


private:
  std::unordered_map<const Val*, StmtNameType> val_map_;
  std::unordered_map<const Expr*, StmtNameType> expr_map_;

  StmtNameType val_name_counter_ = 0;
  StmtNameType expr_name_counter_ = 0;

  StmtNameType getValName() { return val_name_counter_++; }
  StmtNameType getExprName() { return expr_name_counter_++; }

};

// struct TORCH_API Region { // list of nodes (expressions)
// };


//   // fusion: removeNode, addNodeBefore, addNodeAfter, addNodeAt
// };

}}} // torch::jit::fuser

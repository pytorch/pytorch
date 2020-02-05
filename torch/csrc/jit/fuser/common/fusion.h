#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/util/Exception.h>

#include <unordered_map>
#include <vector>
#include <stack>
#include <iostream>
#include <set>

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

/* This file is critical to the lifetime model of the IR system. FusionGuard is a convenient way to set
 * what base container instance holds your IR. Statements that are defined are registered through the FusionGuard
 * with a particular Fusion. You need to create your instances of Fusion and are responsible for holding on to them
 * however, FusionGuard allows you to set this so that constructors of Expr and Vals are automatically registered.
 * If there are other container or derived classes of Statement it is important to make sure it gets registered
 * with Fusion so it is properly cleaned up.
 *
 * Fusion is generally thought of as a translated fusion group from the JIT. It is likely a single kernel, although,
 * we don't have to stick to this in the future and could in theory generate multiple kernels with the logic
 * to execute them.
 *
 * Fusion also allows users to set input/output values that will allow us to figure out how to hook up runtime data
 * to and from the JIT as well as provide us mechanisms for dependency analysis and DCE including safety checks.
 *
 */

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

/*
 * Fusion is mutable but unique. Nodes cannot be copied in any way from one Fusion to another.
 * If anything like that is desired, it would require duplicating all values and exprs.
 * Fusion is considered to contain SSA, though this could also change in the future if there is
 * a good reason to do so.
 *
 * Fusion should provide sorting mechanisms on the Expr's it holds. This could be done in a few
 * ways and we should support as many of them as is convenient. For example we may want to be able
 * to extract out only the expr's that are associated with a particular output, or set of outputs.
 * Fusion should also provide sorted and unsorted iterables of its exprs. This could be either in
 * depth first or breadth first traversal of the dependency chain. Having these sorted lists could
 * make it easy to convert some passes to iteration based instead of recursive, or even allow us to 
 * more easily generate a pass that is partially recursive and partially iterative.
 *
 */

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

  /*
   * Break dependency chains associated with Expr, remove references to expr
   * delete expr.
   */
  void removeExpr(const Expr* expr){
    if(!inFusion(expr))
      throw std::runtime_error("Cannot remove expr as it is not registered with this fusion.");

    for(auto out : expr->outputs())
      if(origin_.find(out) != origin_.end())
        if(origin_.find(out)->second == expr)
          origin_.erase(out);

    for(auto inp : expr->inputs()){
      if(uses_.find(inp) != uses_.end()){
        if(uses_.find(inp)->second.find(expr) != uses_.find(inp)->second.end()){
          uses_.find(inp)->second.erase(expr);
        }
      }
    }

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

  //TODO: Lets topologically sort these.
  std::vector<const Expr*> exprs(bool from_outputs_only=false, bool breadth_first=false) const {
    std::vector<const Expr*> exprs;
    for(const auto it : expr_map_)
      exprs.push_back(it.first);
    return exprs;
  }


  /*
   * Simple Registration methods. These methods do 2 things:
   * Register the Statment/Val/Expr 
   */
  // TODO: Simplify this function, we register values as soon as they're created
  StmtNameType registerVal(const Val* val) {
    if (val->fusion()) {
      TORCH_CHECK(inFusion(val)); //Registered with another fusion
      return val->name();
    }
    val_map_[val] = val->name();
    return val->name();
  }

  /*
   * When we register an expression, we want to update the dependency tracking
   * of Vals. We add expr to our general expr_map_, we add use tracking for inputs
   * and origin tracking for outputs.
   */ 
  StmtNameType registerExpr(const Expr* expr){
    if (expr->fusion()) {
      TORCH_CHECK(inFusion(expr));
      return expr->name();
    }

    expr_map_[expr] = expr->name();

    for (const Val* input : expr->inputs()) {
      registerVal(input);
      if(uses_.find(input) != uses_.end()){
        uses_[input] = {expr};
      }else{
        uses_.find(input)->second.emplace(expr);
      }
    }

    for (const Val* output : expr->outputs()) {
      registerVal(output);
      if(origin_.find(output) != origin_.end()){
        origin_.erase(origin_.find(output));
      }
      origin_[output] = expr;
    }

    return expr->name();
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

  bool used(const Val* val){
    return (uses_.find(val) != uses_.end()) && ( uses_.find(val)->second.size() > 0);
  }

private:
  std::unordered_map<const Val*, StmtNameType> val_map_;
  std::unordered_map<const Expr*, StmtNameType> expr_map_;

  StmtNameType val_name_counter_ = 0;
  StmtNameType expr_name_counter_ = 0;

  StmtNameType getValName() { return val_name_counter_++; }
  StmtNameType getExprName() { return expr_name_counter_++; }

  //Dependency tracking for Vals. Where did it come from? Where is it used?
  std::unordered_map<const Val*, const Expr*> origin_;
  std::unordered_map<const Val*, std::set<const Expr*> > uses_;
};

  ///Convenience methods to be able to print fusions and vals.
  TORCH_API std::ostream& operator<<(std::ostream& os, const Fusion& fusion);
  TORCH_API std::ostream& operator<<(std::ostream& os, const std::deque<const Val*>& vals);

}}} // torch::jit::fuser

#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/util/Exception.h>

#include <unordered_map>
#include <vector>
#include <deque>
#include <iostream>
#include <set>
#include <algorithm>

namespace torch {
namespace jit {
namespace fuser {

// TODO: replaceAllUsesOfLeftWithRight(Val, Val) for values
// TODO: hasUses(Val)
// TODO: plumb through helpers, ex. Val.replaceAllUsesWith(Val)
//          (registration makes this easy)
// TODO: DCE pass

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

    for (auto it = val_set_.begin(); it != val_set_.end(); ++it) {
      delete *it;
    }

    for (auto it = expr_set_.begin(); it != expr_set_.end(); ++it) {
      delete *it;
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

    expr_set_.erase(expr);

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
      infusion &= expr_set_.find(static_cast<const Expr*>(stmt)) != expr_set_.end();
    if(stmt->isVal())
      infusion &= val_set_.find(static_cast<const Val*>(stmt)) != val_set_.end();

    return infusion;
  }

  const Expr* origin(const Val* val) const{
    const auto it = origin_.find(val);

    if(it == origin_.end())
      return nullptr;

    return it->second;
  }

  /*
   * Takes in an unordered map initialized with all Vals to 0, outputs the uses by traversing from
   * output_vals through their dependency chain.
   */
  void populate_uses(std::unordered_map<const Val*, int> &use_count, std::deque<const Val*> outputs) const{
    // outputs is passed by value here.
    std::deque<const Val*> to_visit = outputs; //Could be a stack, but chose deque to match exprs
    std::set<const Val*> visited;
    
    while(!to_visit.empty()){
      const Val* val;
      val = to_visit.back();
      to_visit.pop_back();
      

      if(visited.find(val) != visited.end())
        continue;
      
      visited.emplace(val);
      
      const Expr* orig = origin(val);

      if(orig == nullptr)
        continue;

      for(const Val* inp : orig->inputs()){
        use_count[inp]++;
        // you mean visited.find(inp) instead?
        if(visited.find(inp) != visited.end())
          to_visit.push_back(inp);
      }

      for(const Val* out : orig->outputs())
        if(out != val)
          visited.emplace(out);
    }

  }

  /* 
   * Return topologically sorted list of exprs. We can start by only traversing back from registered
   * outputs, or from any terminating Val. Can also select depth first traversal, or breadth first.
   * 
   * TODO: Test this thing!!!
   */
  std::vector<const Expr*> exprs(bool from_outputs_only=false, bool breadth_first=false) const {

    if(breadth_first)
      throw std::runtime_error("Not implemented yet.");

    std::deque<const Val*> to_visit;  
    std::set<const Val*> visited;

    std::vector<const Expr*> reversed_exprs;

    if(from_outputs_only){
      for(const Val* output : outputs())
        to_visit.push_back(output);
    }else{
      for(const auto it : val_set_){
        if(uses_.find(it) == uses_.end()) //never used, must be output or unused val
          to_visit.push_back(it);
      }
    }

    std::unordered_map<const Val*, int> use_count; //count down to 0, once at 0 all uses have been output, can output origin
    for(const auto it : val_set_){
      use_count[it] = 0;
    }

    populate_uses(use_count, to_visit);

    while(!to_visit.empty()){
      const Val* val;
      if(breadth_first){
        val = to_visit.front();
        to_visit.pop_front();
      }else{
        val = to_visit.back();
        to_visit.pop_back();
      }
      

      // hmm, should we assert here? If we went through a different path to a visited node, that means we screw up the use_count at the first place?
      if(visited.find(val) != visited.end())
        continue;
      
      //0 uses is an output, can't blindly decrement.
      if(use_count[val] != 0){
        use_count[val]--;
        continue;
      }

      const Expr* orig = origin(val);

      //Orig could be nullptr if its an input. We can simply continue in this case
      if(orig == nullptr)
        continue;

      //If other outputs didn't have all uses, can't process expr
      for(const Val* out : orig->outputs())
        if(use_count[out] != 0)
          continue;
          

      //We can visit the origin of val, and mark all outputs of the origin as visited.
      visited.emplace(val);
      for(const Val* out : orig->outputs())
        if(out != val)
          visited.emplace(out);      

      reversed_exprs.push_back(orig);

      for(const Val* inp : orig->inputs())
        if(visited.find(val) != visited.end())
          to_visit.push_back(inp);

    }
    
    std::reverse(reversed_exprs.begin(), reversed_exprs.end());
    return reversed_exprs; //now ordered
  }


  /*
   * Simple Registration methods. These methods do 2 things:
   * Register the Statment/Val/Expr 
   */
  StmtNameType registerVal(const Val* val) {
    if (val->fusion()) {
      TORCH_CHECK(val->fusion() == this);
      if(inFusion(val)) {
        return val->name();
      }
    }
    return getValName();
  }

  /*
   * When we register an expression, we want to update the dependency tracking
   * of Vals. We add expr to our general expr_set_, we add use tracking for inputs
   * and origin tracking for outputs.
   */ 
  StmtNameType registerExpr(const Expr* expr){
    if (expr->fusion()) {
      TORCH_CHECK(expr->fusion() == this);
      if(inFusion(expr)) {
        return expr->name();
      }
    }

    for (const Val* input : expr->inputs()) {
      registerVal(input);
      if(uses_.find(input) == uses_.end()){
        uses_[input] = {expr};
      }else{
        uses_.find(input)->second.emplace(expr);
      }
    }

    for (const Val* output : expr->outputs()) {
      registerVal(output);
      auto it = origin_.find(output);
      if( it != origin_.end()){
        origin_.erase(it);
      }
      
      origin_[output] = expr;
    }

    return getExprName();
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
  std::set<const Val*> val_set_;
  std::set<const Expr*> expr_set_;

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

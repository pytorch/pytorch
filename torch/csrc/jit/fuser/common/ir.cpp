#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/visitor.h>
#include <torch/csrc/jit/fuser/common/mutator.h>

#include <c10/util/Exception.h>

#include <iostream>
#include <unordered_map>
#include <string>
#include <stdexcept>

namespace torch {
namespace jit {
namespace fuser {

/*
* Statement member definitions & related functions
*/

//When we create a Val or EXPR we immediately register them with the active fusion.
Val::Val(
  const ValType _type) 
  : type_{_type} {
    Fusion* fusion = FusionGuard::getCurFusion(); 
    if( fusion != nullptr){
      this->name_ = fusion->registerVal(this);
      this->fusion_ = fusion;
    }else{
      throw std::runtime_error("No fusion group found when creating a Val.");
    }
    
}


Expr::Expr(
    const ExprType _type)
  : type_{_type} {
    Fusion* fusion = FusionGuard::getCurFusion(); 
    if(fusion == nullptr)
      throw std::runtime_error("No fusion group found when creating an Expr.");
    this->fusion_ = fusion;
}

  Add::Add(
    const Val* _out
  , const Val* _lhs
  , const Val* _rhs)
  : Expr(ExprType::Add)
  , out_{_out}
  , lhs_{_lhs}
  , rhs_{_rhs} {
    addOutput(_out);
    addInput(_lhs);
    addInput(_rhs);
    this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
  }

Statement::~Statement() { }

template <typename T>
T* ptr(T& obj) { return &obj; }

template <typename T>
T* ptr(T* obj) { return obj; }


/*
 * Generic dispatch for any handler that does not modify the IR directly. 
 * For example we may want to walk the graph to construct a topologically sorted
 * set of exprs. This doesn't modify the IR directly. We also use this to print
 * the IR itself.
 * This dispatch is paired with a class that implements the functions:
 * template <typenname node_type>
 * int handler(const node_type* node)
 *
 * handler should call:
 * (statement* node_to_dispatch)->dispatch()
 *
 * It could also implement:
 * int handler(Statement* stmt){
 *   stmt->dispatch(this); 
 * }
 *
 * And therefore dispatch should never call:
 * ptr(mutator)->handle(static_cast<const Statement*>(this));
 */
 

template <typename T>
int Statement::dispatch(T handler) const{
  if (isVal()) {
    switch (*getValType()) {
      case ValType::Tensor:
        return ptr(handler)->handle(static_cast<const Tensor*>(this));
      case ValType::Float:
        return ptr(handler)->handle(static_cast<const Float*>(this));
      case ValType::Int:
        return ptr(handler)->handle(static_cast<const Int*>(this));
      default:
        throw std::runtime_error("Unknown valtype in dispatch!");
    }
  }

  if(isExpr()){
    switch (*getExprType()) {
      case ExprType::Add:
        return ptr(handler)->handle(static_cast<const Add*>(this));
      default:
        throw std::runtime_error("Unknown exprtype in dispatch!");
    }
  }

  throw std::runtime_error("Unknown stmttype in dispatch!");

}

/*
 * Generic dispatch for any handler that modifies the IR. This could be a transformation
 * on loop structures, or parallelizing a loop.
 * This dispatch is paired with a class that implements the functions
 * template <typenname node_type>
 * const Statement* mutate(const node_type* node)
 * mutate should call (statement* node_to_dispatch)->dispatch_mutator()
 * It could also implement
 * const Statement* mutate(Statement* stmt){
 *   stmt->dispatch_mutator(this); 
 * }
 * And therefore dispatch_mutator should never call:
 *   ptr(mutator)->mutate(static_cast<const Statement*>(this));
 */
//NEVER CALL MUTATE ON A CONST STATEMENT* FROM HERE!
//otherwise you'll end in an infinite loop with mutate.
template <typename T>
const Statement* Statement::dispatch_mutator(T mutator) const{
  if (isVal()) {
    switch (*getValType()) {
      case ValType::Tensor:
        return ptr(mutator)->mutate(static_cast<const Tensor*>(this));
      case ValType::Float:
        return ptr(mutator)->mutate(static_cast<const Float*>(this));
      case ValType::Int:
        return ptr(mutator)->mutate(static_cast<const Int*>(this));
      default:
        throw std::runtime_error("Unknown valtype in dispatch_mutator!");
    }
  }

  if(isExpr()){
    switch (*getExprType()) {
      case ExprType::Add:
        return ptr(mutator)->mutate(static_cast<const Add*>(this));
      default:
        throw std::runtime_error("Unknown exprtype in dispatch_mutator!");
    }
  }
  throw std::runtime_error("Unknown stmttype in dispatch_mutator!");
}


// Handler template instantiations
template const Statement* Statement::dispatch_mutator(BaseMutator) const;
template const Statement* Statement::dispatch_mutator(BaseMutator*) const;

/*
* Val member definitions
*/

Val::~Val() { }

/*
* IRInputOutput member definitions
*/

IRInputOutput::~IRInputOutput() { }

/*
* Expr member definitions
*/

Expr::~Expr() { }


}}} // torch::jit::fuser

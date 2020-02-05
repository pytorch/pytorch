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

/*
void Val::setOrigin(Expr* new_origin) const{
    if(new_origin == origin_)
      return;

    assert(new_origin->isExpr());
    assert(new_origin->isAnOutput(this));

    Expr* old_origin = origin_;
    origin_ = new_origin;

    if(old_origin!=nullptr){
      old_origin->removeOutput(this);
      if(old_origin->nOutputs() == 1){
        FusionGuard::getCurFusion()->removeExpr(old_origin);
      }
    }
  }
*/

Expr::Expr(
    const ExprType _type)
  : type_{_type} {
    Fusion* fusion = FusionGuard::getCurFusion(); 
    if(fusion == nullptr)
      throw std::runtime_error("No fusion group found when creating an Expr.");
    this->name_ = fusion->registerExpr(this);
    this->fusion_ = fusion;
    
    //for(const Val* output : outputs_)
    //  output->setOrigin(this);
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
  const auto maybe_val_type = getValType();
  if (maybe_val_type) {
    switch (*maybe_val_type) {
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

  switch (*getExprType()) {
    case ExprType::Add:
      return ptr(handler)->handle(static_cast<const Add*>(this));
    default:
      throw std::runtime_error("Unknown exprtype in dispatch!");
  }
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
  const auto maybe_val_type = getValType();
  if (maybe_val_type) {
    switch (*maybe_val_type) {
      case ValType::Tensor:
        return ptr(mutator)->mutate(static_cast<const Tensor*>(this));
      case ValType::Float:
        return ptr(mutator)->mutate(static_cast<const Float*>(this));
      case ValType::Int:
        return ptr(mutator)->mutate(static_cast<const Int*>(this));
      default:
        throw std::runtime_error("Unknown valtype in dispatch!");
    }
  }

  switch (*getExprType()) {
    case ExprType::Add:
      return ptr(mutator)->mutate(static_cast<const Add*>(this));
    default:
      throw std::runtime_error("Unknown exprtype in dispatch!");
  }
}


// Handler template instantiations
template int Statement::dispatch(SimpleHandler) const;
template int Statement::dispatch(SimpleHandler*) const;
template int Statement::dispatch(IRPrinter) const;
template int Statement::dispatch(IRPrinter*) const;

template const Statement* Statement::dispatch_mutator(BaseMutator) const;
template const Statement* Statement::dispatch_mutator(BaseMutator*) const;

std::ostream& operator<<(std::ostream& out, const Statement* const stmt) {
  IRPrinter printer{out};
  stmt->dispatch(printer);
  return out;
}

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

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

Val::Val(
  const ValType _type) 
  : type_{_type} {
    Fusion* fusion = FusionGuard::getCurFusion(); 
    if( fusion != nullptr){
      fusion->registerVal(this);
      this->fusion_ = fusion;
    }else{
      throw std::runtime_error("No fusion group found when creating a Val.");
    }
    
}

Expr::Expr(
    const ExprType _type)
  : type_{_type} {
    Fusion* fusion = FusionGuard::getCurFusion(); 
    if(fusion != nullptr){
      fusion->registerExpr(this);
      this->fusion_ = fusion;
    }else{
      throw std::runtime_error("No fusion group found when creating an Expr.");
    }
}

Statement::~Statement() { }

template <typename T>
T* ptr(T& obj) { return &obj; }

template <typename T>
T* ptr(T* obj) { return obj; }

// Note: when adding a new val or expr a case must be added here
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

// void Expr::addRegion(Region* region) {
//   if (region->parent()) {
//     TORCH_CHECK(region->parent() == this);
//     return;
//   }

//   region->setParent(this);
//   if (fusion_) {
//     fusion_->registerRegion(region);
//   }

//   regions_.push_back(region);
// }

// void Expr::register_callback(Statement* stmt) {
//   if (region_) {
//     region_->registerStatement(stmt);
//   } else if (fusion_) {
//     fusion_->registerStatement(stmt);
//   }
// }

}}} // torch::jit::fuser

#pragma once
#include <ostream>
#include "IRVisitor.h"

namespace Fuser{

template<typename T>
class FindInstances : public IRVisitor {
public:
  void visit(const T* op){
    instances.push_back(op);
    IRVisitor::visit(op);
  }
  std::vector<Expr> instances;

};

template<typename T>
std::vector<Expr> findAll(Expr expr){
  FindInstances<T> finder;
  expr.accept(&finder);
  return finder.instances;
}

}//namespace Fuser
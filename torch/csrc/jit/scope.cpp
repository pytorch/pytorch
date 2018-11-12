#include "ir.h"


#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/assertions.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <stack>
#include <sstream>
#include <algorithm>
#include <string>

namespace torch { namespace jit {

ScopePtr Scope::push(Symbol name) {
  return c10::make_intrusive<Scope>(intrusive_from_this(), name);
}

ScopePtr Scope::getRoot() {
  ScopePtr current = intrusive_from_this();
  while (current->parent_) {
    current = current->parent_;
  }
  return current;
}

size_t Scope::getDepth() {
  size_t d = 1;
  ScopePtr current = intrusive_from_this();
  while (current->parent_) {
    current = current->parent_;
    d += 1;
  }
  return d;
}

std::string Scope::namesFromRoot(const std::string& separator) const {
  // TODO: I think the answer is we shouldn't have used Symbol here
  std::string out = this->name_.toUnqualString();
  if (this->isRoot()) {
    return out;
  }
  ScopePtr parent = this->parent_;
  while (!parent->isRoot()) {
    // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
    out = std::string(parent->name_.toUnqualString()) + separator + out;
    parent = parent->parent_;
  }
  return out;
}

}} // namespace torch::jit

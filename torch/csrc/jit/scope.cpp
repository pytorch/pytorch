#include <torch/csrc/jit/scope.h>
#include <torch/csrc/jit/function.h>

namespace torch {
namespace jit {

ScopePtr Scope::intrusive_from_this() {
  c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                         // from a raw `this` pointer
                                         // so we need to bump the refcount
                                         // to account for this ownership
  return c10::intrusive_ptr<Scope>::reclaim(this);
}

Scope::Scope() {
  name_ = Symbol::scope("");
}

Scope::Scope(ScopePtr parent, Symbol name) {
  name_ = name;
  parent_ = std::move(parent);
}

ScopePtr Scope::push(Symbol name) {
  return c10::make_intrusive<Scope>(intrusive_from_this(), name);
}

ScopePtr Scope::parent() {
  if (!parent_) {
    throw std::runtime_error("Cannot get parent from Scope with no parent");
  }
  return parent_;
}

bool Scope::isRoot() const {
  return !parent_;
}

bool Scope::isBlank() const {
  static const Symbol blank = Symbol::scope("");
  return isRoot() && name() == blank;
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

Symbol Scope::name() const {
  return name_;
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

CallStackPtr CallStack::intrusive_from_this() {
  c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                         // from a raw `this` pointer
                                         // so we need to bump the refcount
                                         // to account for this ownership
  return c10::intrusive_ptr<CallStack>::reclaim(this);
}

CallStack::CallStack(Function* fn) : fn_(fn) {}

CallStack::CallStack(CallStackPtr caller, Function* fn) : fn_(fn) {
  caller_ = std::move(caller);
}

CallStackPtr CallStack::insertCallee(Function* fn) {
  if (callees_.count(fn)) {
    return callees_.at(fn);
  }
  auto subscope = c10::make_intrusive<CallStack>(intrusive_from_this(), fn);
  callees_[fn] = subscope;
  return subscope;
}

c10::optional<CallStackPtr> CallStack::caller() const {
  return caller_;
}

std::vector<Function*> CallStack::asVector() {
  std::vector<Function*> r;
  c10::optional<CallStackPtr> current = intrusive_from_this();
  while (current) {
    r.push_back((*current)->fn_);
    current = (*current)->caller_;
  }
  return std::vector<Function*>(r.rbegin(), r.rend());
}

} // namespace jit
} // namespace torch

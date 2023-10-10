#include <torch/csrc/jit/ir/scope.h>

#include <ATen/core/class_type.h>
#include <ATen/core/function.h>

namespace torch::jit {
// util functions
namespace utils {

std::string get_module_info(const ModuleInstanceInfo& module_instance_info) {
  std::string module_info;
  const auto& class_type = module_instance_info.class_type();
  std::string instance_name = module_instance_info.instance_name();
  std::string type_name;
  if (class_type) {
    type_name += class_type->name()->qualifiedName();
    type_name = type_name.substr(type_name.find_last_of('.') + 1);
  }
  if (type_name.empty()) {
    type_name = "UNKNOWN_TYPE";
  }
  if (instance_name.empty()) {
    instance_name = "UNKNOWN_INSTANCE";
  }
  module_info.append(instance_name).append("(").append(type_name).append(")");
  return module_info;
}

} // namespace utils
ScopePtr Scope::intrusive_from_this() {
  c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                         // from a raw `this` pointer
                                         // so we need to bump the refcount
                                         // to account for this ownership
  return c10::intrusive_ptr<Scope>::reclaim(this);
}

Scope::Scope() : name_(Symbol::scope("")) {}

Scope::Scope(ScopePtr parent, Symbol name)
    : parent_(std::move(parent)), name_(name) {}

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

InlinedCallStackPtr InlinedCallStack::intrusive_from_this() {
  c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                         // from a raw `this` pointer
                                         // so we need to bump the refcount
                                         // to account for this ownership
  return c10::intrusive_ptr<InlinedCallStack>::reclaim(this);
}

InlinedCallStack::InlinedCallStack(Function* fn, SourceRange source_range)
    : fn_(fn),
      fn_name_(fn_ ? fn_->name() : ""),
      source_range_(std::move(source_range)) {}

InlinedCallStack::InlinedCallStack(
    Function* fn,
    SourceRange source_range,
    c10::optional<ModuleInstanceInfo> module_instance_info)
    : fn_(fn),
      fn_name_(fn_ ? fn_->name() : ""),
      source_range_(std::move(source_range)),
      module_instance_info_(std::move(module_instance_info)) {}

InlinedCallStack::InlinedCallStack(
    Function* fn,
    SourceRange source_range,
    c10::optional<ModuleInstanceInfo> module_instance_info,
    std::string& function_name)
    : fn_(fn),
      fn_name_(std::move(function_name)),
      source_range_(std::move(source_range)),
      module_instance_info_(std::move(module_instance_info)) {}

InlinedCallStack::InlinedCallStack(
    InlinedCallStackPtr callee,
    Function* fn,
    SourceRange source_range)
    : callee_(std::move(callee)),
      fn_(fn),
      fn_name_(fn_ ? fn_->name() : ""),
      source_range_(std::move(source_range)) {}

InlinedCallStack::InlinedCallStack(
    InlinedCallStackPtr callee,
    Function* fn,
    SourceRange source_range,
    c10::optional<ModuleInstanceInfo> module_instance_info,
    std::string& function_name)
    : callee_(std::move(callee)),
      fn_(fn),
      fn_name_(std::move(function_name)),
      source_range_(std::move(source_range)),
      module_instance_info_(std::move(module_instance_info)) {}

InlinedCallStack::InlinedCallStack(
    InlinedCallStackPtr callee,
    Function* fn,
    SourceRange source_range,
    c10::optional<ModuleInstanceInfo> module_instance_info)
    : callee_(std::move(callee)),
      fn_(fn),
      fn_name_(fn_ ? fn_->name() : ""),
      source_range_(std::move(source_range)),
      module_instance_info_(std::move(module_instance_info)) {}

c10::optional<InlinedCallStackPtr> InlinedCallStack::callee() const {
  return callee_;
}

void InlinedCallStack::setCallee(c10::optional<InlinedCallStackPtr> callee) {
  callee_ = std::move(callee);
}

c10::optional<ModuleInstanceInfo> InlinedCallStack::module_instance() const {
  return module_instance_info_;
}

SourceRange InlinedCallStack::source_range() const {
  return source_range_;
}

Function* InlinedCallStack::function() const {
  return fn_;
}

const std::string& InlinedCallStack::function_name() const {
  return fn_name_;
}

std::vector<InlinedCallStackEntry> InlinedCallStack::vec() {
  std::vector<InlinedCallStackEntry> r;
  c10::optional<InlinedCallStackPtr> current = intrusive_from_this();
  while (current) {
    r.emplace_back(
        (*current)->fn_,
        (*current)->source_range_,
        (*current)->module_instance_info_);
    current = (*current)->callee_;
  }
  return r;
}

ModuleInstanceInfo::ModuleInstanceInfo(
    c10::ClassTypePtr module_type,
    std::string instance_name)
    : module_type_(std::move(module_type)),
      instance_name_(std::move(instance_name)) {}
} // namespace torch::jit

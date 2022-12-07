#include <c10/util/Exception.h>
#include <operator_registry.h>

namespace torch {
namespace executor {

OperatorRegistry& getOperatorRegistry() {
  static OperatorRegistry operator_registry;
  return operator_registry;
}

bool register_operators(const ArrayRef<Operator>& operators) {
  return getOperatorRegistry().register_operators(operators);
}

bool OperatorRegistry::register_operators(
    const ArrayRef<Operator>& operators) {
  for (const auto& op : operators) {
    this->operators_map_[op.name_] = op.op_;
  }
  return true;
}

bool hasOpsFn(const char* name) {
  return getOperatorRegistry().hasOpsFn(name);
}

bool OperatorRegistry::hasOpsFn(const char* name) {
  auto op = this->operators_map_.find(name);
  return op != this->operators_map_.end();
}

OpFunction& getOpsFn(const char* name) {
  return getOperatorRegistry().getOpsFn(name);
}

OpFunction& OperatorRegistry::getOpsFn(const char* name) {
  auto op = this->operators_map_.find(name);
  TORCH_CHECK_MSG(op != this->operators_map_.end(), "Operator not found!");
  return op->second;
}


} // namespace executor
} // namespace torch

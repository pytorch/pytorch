#pragma once

#include <cstring>

#include <c10/ArrayRef.h>
#include <c10/IValue.h>
#include <functional>
#include <map>

namespace torch {
namespace executor {


using EValue = torch::IValue;
using OpFunction = std::function<void(EValue**)>;
struct Operator {
  const char* name_;
  OpFunction op_;

  Operator() = default;

  /**
   * We are doing a copy of the string pointer instead of duplicating the string
   * itself, we require the lifetime of the operator name to be at least as long
   * as the operator registry.
   */
  explicit Operator(const char* name, OpFunction func)
      : name_(name), op_(func) {}
};

/**
 * See OperatorRegistry::hasOpsFn()
 */
bool hasOpsFn(const char* name);

/**
 * See OperatorRegistry::getOpsFn()
 */
OpFunction& getOpsFn(const char* name);

/**
 * See OperatorRegistry::getOpsArray()
 */
ArrayRef<Operator> getOpsArray();


void register_operators(const ArrayRef<Operator>&);

struct OperatorRegistry {
 public:
  OperatorRegistry() : operatorRegSize_(0) {}

  void register_operators(const ArrayRef<Operator>&);

  /**
   * Checks whether an operator with a given name is registered
   */
  bool hasOpsFn(const char* name);

  /**
   * Checks whether an operator with a given name is registered
   */
  OpFunction& getOpsFn(const char* name);

 private:
  std::map<const char*, OpFunction> operators_map_;
  uint32_t operatorRegSize_;
};

} // namespace executor
} // namespace torch

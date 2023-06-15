#pragma once

#include <cstring>

#include <c10/util/ArrayRef.h>
#include "Evalue.h"
#include "RuntimeContext.h"
#include <functional>
#include <map>

namespace torch {
namespace executor {

using OpFunction = std::function<void(RuntimeContext&, EValue**)>;

template<typename T>
using ArrayRef = at::ArrayRef<T>;

#define EXECUTORCH_SCOPE_PROF(x)

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


[[nodiscard]] bool register_operators(const ArrayRef<Operator>&);

struct OperatorRegistry {
 public:
  OperatorRegistry() : operatorRegSize_(0) {}

  bool register_operators(const ArrayRef<Operator>&);

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

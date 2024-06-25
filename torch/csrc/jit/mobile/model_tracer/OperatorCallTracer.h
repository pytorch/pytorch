#pragma once

#include <ATen/record_function.h>
#include <c10/util/Synchronized.h>

namespace torch {
namespace jit {
namespace mobile {
/* The OperatorCallTracer class handles the attachment and removal of a
 * recording callback that traces invocation of ATen (and other) PyTorch
 * operators that get called via the Dispatcher.
 *
 * You can get the set of operators that were called (op_name.overload_name)
 * using getCalledOperators().
 *
 * Note: This class is not thread safe or re-entrant, and should not be used
 * across multiple threads of execution.
 *
 */
struct OperatorCallTracer final {
  at::CallbackHandle handle_;

  OperatorCallTracer();

  static c10::Synchronized<std::set<std::string>>& getCalledOperators() {
    static c10::Synchronized<std::set<std::string>> called_operators_;
    return called_operators_;
  }

  ~OperatorCallTracer() {
    at::removeCallback(handle_);
  }
};
} // namespace mobile
} // namespace jit
} // namespace torch

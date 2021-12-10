#pragma once

#include <ATen/record_function.h>

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
  static std::set<std::string> called_operators_;
  at::CallbackHandle handle_;

  OperatorCallTracer() {
    called_operators_.clear();
    auto recorder_cb = [](const at::RecordFunction& fn)
        -> std::unique_ptr<at::ObserverContext> {
      c10::optional<c10::OperatorName> op_name = fn.operator_name();
      if (op_name.has_value()) {
        called_operators_.insert(c10::toString(*op_name));
      }
      return nullptr;
    };

    handle_ = at::addGlobalCallback(at::RecordFunctionCallback(recorder_cb)
                                        .scopes({at::RecordScope::FUNCTION}));
  }

  std::set<std::string> const& getCalledOperators() const {
    return called_operators_;
  }

  ~OperatorCallTracer() {
    at::removeCallback(handle_);
  }
};
} // namespace mobile
} // namespace jit
} // namespace torch

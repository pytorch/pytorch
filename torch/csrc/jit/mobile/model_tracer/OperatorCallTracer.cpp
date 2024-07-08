#include <torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.h>

namespace torch {
namespace jit {
namespace mobile {
OperatorCallTracer::OperatorCallTracer() {
  getCalledOperators().withLock([](std::set<std::string>& called_operators) {
    called_operators.clear();
  });

  auto recorder_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    std::optional<c10::OperatorName> op_name = fn.operator_name();
    if (op_name.has_value()) {
      getCalledOperators().withLock(
          [op_name](std::set<std::string>& called_operators) {
            called_operators.insert(c10::toString(*op_name));
          });
    }
    return nullptr;
  };

  handle_ = at::addGlobalCallback(at::RecordFunctionCallback(recorder_cb)
                                      .scopes({at::RecordScope::FUNCTION}));
}

} // namespace mobile
} // namespace jit
} // namespace torch

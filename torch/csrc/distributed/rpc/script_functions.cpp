#include <torch/csrc/distributed/rpc/script_functions.h>

#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

c10::IValue rpcTorchscriptCall(
    const std::string& dst,
    const c10::QualifiedName& qualifiedName,
    std::vector<c10::IValue>& stack) {
  auto scriptCall =
      c10::guts::make_unique<ScriptCall>(qualifiedName, std::move(stack));
  auto agent = RpcAgent::getDefaultRpcAgent();
  auto futMessage = autograd::sendMessageWithAutograd(
      *agent, agent->getWorkerInfo(dst), std::move(*scriptCall).toMessage());
  // Get function return type to construct c10::ivalue::Future.
  // Script call only allows single IValue returned.
  auto returns = torch::jit::get_python_cu()
                     ->get_function(qualifiedName)
                     .getSchema()
                     .returns();
  TORCH_INTERNAL_ASSERT(
      returns.size() == 1,
      "Return value of a annotated torchScript function should be a single "
      "IValue, got a vector of size ",
      returns.size());
  auto returnType = returns.at(0).type();

  // Create a JIT future and pass it to futMessage's callback to set state
  // of the JIT future.
  auto futPtr = c10::make_intrusive<c10::ivalue::Future>(returnType);
  futMessage->addCallback([futPtr](
                              const Message& message,
                              bool hasError,
                              const utils::FutureError& futErr) {
    if (hasError) {
      throw std::runtime_error(futErr.errMsg());
    }
    futPtr->markCompleted(deserializeRespToIValue(message));
  });
  return IValue(futPtr);
}

} // namespace rpc
} // namespace distributed
} // namespace torch

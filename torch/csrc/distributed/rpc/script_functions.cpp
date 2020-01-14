#include <torch/csrc/distributed/rpc/script_functions.h>

#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

c10::intrusive_ptr<c10::ivalue::Future> rpcTorchscriptCall(
    const std::string& dst,
    const c10::QualifiedName& qualifiedName,
    std::vector<c10::IValue>& stack) {
  auto scriptCall =
      std::make_unique<ScriptCall>(qualifiedName, std::move(stack));
  auto agent = RpcAgent::getDefaultRpcAgent();
  auto futMessage = autograd::sendMessageWithAutograd(
      *agent, agent->getWorkerInfo(dst), std::move(*scriptCall).toMessage());
  // Get function return type to construct c10::ivalue::Future.
  // Script call only allows single IValue returned.
  auto returns = PythonRpcHandler::getInstance()
                     .jitCompilationUnit()
                     ->get_function(qualifiedName)
                     .getSchema()
                     .returns();
  TORCH_INTERNAL_ASSERT(
      returns.size() == 1,
      "Return value of an annotated torchScript function should be a single "
      "IValue.",
      returns.size());
  auto returnType = returns.at(0).type();

  // Create a JIT future and pass it to futMessage's callback to set state
  // of the JIT future.
  auto futPtr = c10::make_intrusive<c10::ivalue::Future>(returnType);
  futMessage->addCallback([futPtr](
                              const rpc::Message& message,
                              const c10::optional<utils::FutureError>& futErr) {
    if (futErr) {
      c10::ivalue::Future::FutureError jitFutErr(std::string((*futErr).what()));
      futPtr->markCompleted(std::move(jitFutErr));
    } else {
      futPtr->markCompleted(deserializeRespToIValue(message));
    }
  });
  return futPtr;
}

} // namespace rpc
} // namespace distributed
} // namespace torch

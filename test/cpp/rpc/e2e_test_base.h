#include <gtest/gtest.h>

#include <c10d/TCPStore.hpp>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace distributed {
namespace rpc {

using torch::distributed::autograd::DistAutogradContainer;
using torch::distributed::autograd::DistAutogradContext;

DistAutogradContainer* getDistAutogradContainer();

class TestE2EBase : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup distributed autograd.
    autogradContainer = getDistAutogradContainer();

    // Setup server store.
    store = std::make_shared<c10d::TCPStore>(
        serverAddress, 0, numWorkers, true, std::chrono::seconds(10));

    buildRpcAgent();

    rpcAgentPostProcessing();
  }

  void rpcAgentPostProcessing() {
    RpcAgent::setCurrentRpcAgent(rpcAgent);
    std::shared_ptr<TypeResolver> typeResolver =
        std::make_shared<TypeResolver>([&](const c10::QualifiedName& qn) {
          return c10::StrongTypePtr(
              nullptr, c10::TensorType::create(at::Tensor()));
        });
    rpcAgent->setTypeResolver(typeResolver);
    rpcAgent->start();
  }

  void TearDown() override {
    rpcAgent->join();
    rpcAgent->shutdown();
    RpcAgent::setCurrentRpcAgent(nullptr);
  }

  c10::intrusive_ptr<OwnerRRef> createRemoteRRef(
      at::Tensor t1,
      at::Tensor t2,
      std::shared_ptr<torch::jit::Operator> op) {
    auto& ctx = RRefContext::getInstance();
    auto ownerRRef = ctx.createOwnerRRef(c10::TensorType::create(t1));
    // prevent this owner RRef being deleted due to other forks
    ctx.addSelfAsFork(ownerRRef);

    ScriptRemoteCall scriptRemoteCall(
        op, {t1, t2, 1}, ownerRRef->rrefId(), ownerRRef->rrefId());
    auto fm = autograd::sendMessageWithAutograd(
        *rpcAgent,
        rpcAgent->getWorkerInfo("worker"),
        std::move(scriptRemoteCall).toMessage(),
        false);

    ownerRRef->registerOwnerCreationFuture(fm);

    // Builtin operators does not return py::object, and hence does not require
    // GIL for destructing the potentially deleted OwerRRef.
    fm->addCallback(
        [ownerRRefId = ownerRRef->rrefId()](const FutureMessage& fm) {
          callback::finishCreatingOwnerRRef(fm, ownerRRefId);
        });
    return ownerRRef;
  }

  at::Tensor remoteAdd(
      at::Tensor t1,
      at::Tensor t2,
      std::shared_ptr<torch::jit::Operator> op) {
    ScriptCall scriptCall(op, {t1, t2, /* alpha */ 1});

    // Send the RPC and return result.
    auto response = autograd::sendMessageWithAutograd(
                        *rpcAgent,
                        rpcAgent->getWorkerInfo("worker"),
                        std::move(scriptCall).toMessage())
                        ->wait();
    MessageType messageType = MessageType::FORWARD_AUTOGRAD_RESP;
    auto wrappedResponse = deserializeResponse(response, messageType);
    return static_cast<ScriptResp&>(*wrappedResponse).value().toTensor();
  }

  virtual void buildRpcAgent() = 0;

  class AutogradContextGuard {
   public:
    explicit AutogradContextGuard()
        : context(DistAutogradContainer::getInstance().newContext()) {}

    ~AutogradContextGuard() {
      DistAutogradContainer::getInstance().releaseContext(context->contextId());
    }

   private:
    std::shared_ptr<DistAutogradContext> context;
  };

  void runTrainingLoop() {
    auto options = at::TensorOptions().requires_grad(true);
    auto t1 = torch::ones({3, 3}, options);
    auto t2 = torch::ones({3, 3}, options);

    c10::OperatorName full_name("aten::add", "Tensor");
    auto matchedOp = torch::jit::findOperatorFor(full_name);
    ASSERT_TRUE(matchedOp);

    for (size_t i = 0; i < numIters; i++) {
      // Create the autograd context guard.
      AutogradContextGuard guard;

      // Multiple RPCs within one autograd context for the forward pass.
      auto result = remoteAdd(t1, t2, matchedOp);
      for (size_t j = 0; j < 5; j++) {
        result = remoteAdd(t1, result, matchedOp);
      }

      auto rref = createRemoteRRef(t1, result, matchedOp);
      result = rref->getValue().toTensor();

      // Run backward pass now.
      autograd::DistEngine::getInstance().execute(
          DistAutogradContainer::currentContextId(),
          {torch::sum(result)},
          /* retainGraph */ false);
    }
  }

  DistAutogradContainer* autogradContainer;
  std::shared_ptr<RpcAgent> rpcAgent;
  static const size_t numIters;
  static const size_t numWorkers;
  std::shared_ptr<c10d::Store> store;
  static const char* serverAddress;
};

} // namespace rpc
} // namespace distributed
} // namespace torch

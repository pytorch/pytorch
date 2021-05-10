#include <memory>

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/torch.h>

namespace torch {
namespace distributed {
namespace autograd {

class DistAutogradTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    autogradContainer_ = &DistAutogradContainer::init(0);
  }

  virtual void TearDown() {
    autogradContainer_->releaseContext(autogradContainer_->currentContext()->contextId());
  }

  static DistAutogradContainer* autogradContainer_;
};

DistAutogradContainer* DistAutogradTest::autogradContainer_ = nullptr;

TEST_F(DistAutogradTest, TestSendFunctionInvalidInputs) {
  auto options = at::TensorOptions().requires_grad(true);
  auto in1 = torch::ones({3, 3}, options);
  auto in2 = torch::ones({3, 3}, options);

  autogradContainer_->newContext();
  auto autogradContext = autogradContainer_->currentContext();
  // Attach the send autograd function to tensors.
  std::vector<torch::Tensor> tensors = {in1, in2};
  rpc::worker_id_t worker_id = 1;
  addSendRpcBackward(
      autogradContext, AutogradMetadata(1, 1), tensors);
  autogradContext->addKnownWorkerId(worker_id);
  auto send_function = autogradContext->sendFunctions()[1];

  // ensure that the worker_ids are recorded
  auto knownWorkerIds = autogradContext->getKnownWorkerIds();
  ASSERT_TRUE(knownWorkerIds.find(worker_id) != knownWorkerIds.end());
  ASSERT_EQ(knownWorkerIds.size(), 1);

  // This should fail since the SendRpcBackward function shouldn't receive any
  // inputs grad.
  EXPECT_THROW(send_function->apply({in1, in2}), c10::Error);

  // This should fail since the SendRpcBackward function encounters an undefined
  // grad.
  send_function->setGrads({in1, torch::autograd::Variable()});
  EXPECT_THROW(send_function->apply({}), c10::Error);
}

TEST_F(DistAutogradTest, TestInitializedContextCleanup) {
  autogradContainer_->newContext();
  auto contextId = autogradContainer_->currentContext()->contextId();
  auto& engine = DistEngine::getInstance();
  ASSERT_EQ(0, engine.numBackwardPasses());

  // Build autograd graph
  auto x = torch::randn({2, 2}, torch::requires_grad());
  auto y = torch::randn({2, 2}, torch::requires_grad());
  auto z = (x * x + y * y).sum();
  ASSERT_NE(nullptr, z.grad_fn());

  // Execute engine.
  engine.execute(contextId, {z}, /* retainGraph */ false);

  // Validate appropriate cleanup.
  ASSERT_EQ(0, engine.numBackwardPasses());
}

TEST_F(DistAutogradTest, TestInitializedContextCleanupSendFunction) {
  autogradContainer_->newContext();
  auto context = autogradContainer_->currentContext();
  auto& engine = DistEngine::getInstance();
  ASSERT_EQ(0, engine.numBackwardPasses());

  // Attach send function.
  auto options = at::TensorOptions().requires_grad(true);
  auto t = torch::ones({1}, options);
  auto tensors = std::vector<torch::Tensor>{t};
  addSendRpcBackward(
      context, AutogradMetadata(context->contextId(), 0), tensors);

  auto sendFunction = context->retrieveSendFunction(0);
  sendFunction->setGrads({t});

  // Execute engine.
  engine
      .executeSendFunctionAsync(context, sendFunction, /*retrainGraph*/ false)
      ->wait();

  // Validate appropriate cleanup.
  ASSERT_EQ(0, engine.numBackwardPasses());
}

} // namespace autograd
} // namespace distributed
} // namespace torch

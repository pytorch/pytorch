#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/torch.h>

using namespace torch::distributed::autograd;
using namespace torch::distributed::rpc;

class DistAutogradTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    autogradContainer_ = &DistAutogradContainer::init(0);
  }
  static DistAutogradContainer* autogradContainer_;
};

DistAutogradContainer* DistAutogradTest::autogradContainer_ = nullptr;

TEST_F(DistAutogradTest, TestSendFunctionInvalidInputs) {
  auto options = at::TensorOptions().requires_grad(true);
  auto in1 = torch::ones({3, 3}, options);
  auto in2 = torch::ones({3, 3}, options);

  autogradContainer_->newContext();
  DistAutogradContext& autogradContext = autogradContainer_->currentContext();
  // Attach the send autograd function to tensors.
  std::vector<torch::Tensor> tensors = {in1, in2};
  worker_id_t worker_id = 1;
  addSendRpcBackward(
      autogradContext, AutogradMetadata(1, 1), tensors, worker_id);
  auto send_function = autogradContext.sendFunctions()[1];

  // ensure that the worker_ids are recorded
  auto knownWorkerIds = autogradContext.getKnownWorkerIds();
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

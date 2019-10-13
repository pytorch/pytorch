#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_context.h>
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

  // This should fail since the SendRpcBackward function shouldn't receive any
  // inputs grad.
  EXPECT_THROW(send_function->apply({in1, in2}), c10::Error);

  // This should fail since the SendRpcBackward function encounters an undefined
  // grad.
  send_function->setGrads({in1, torch::autograd::Variable()});
  EXPECT_THROW(send_function->apply({}), c10::Error);
}

TEST_F(DistAutogradTest, TestWorkerIdsRecorded) {
  auto options = at::TensorOptions().requires_grad(false);
  auto in1 = torch::ones({3, 3}, options);
  auto in2 = torch::ones({3, 3}, options);

  autogradContainer_->newContext();
  DistAutogradContext& autogradContext =
      autogradContainer_->currentContext();
  std::vector<torch::Tensor> tensors = {in1, in2};

  // ensure that if we do not add the send function, then we don't record the
  // worker id
  worker_id_t dst_no_grad = 1;
  addSendRpcBackward(
      autogradContext, AutogradMetadata(1, 1), tensors, dst_no_grad);
  auto knownWorkerIds = autogradContext.getKnownWorkerIds();
  ASSERT_TRUE(knownWorkerIds.find(dst_no_grad) == knownWorkerIds.end());

  // when the tensors do require grad, we will attach the send function. Make
  // sure that the workerId is recorded.

  in1.set_requires_grad(true);
  in2.set_requires_grad(true);
      autogradContainer_->currentContext();
  worker_id_t dst_grad = 2;
  addSendRpcBackward(
      autogradContext, AutogradMetadata(1, 1), tensors, dst_grad);
  knownWorkerIds = autogradContext.getKnownWorkerIds();
  ASSERT_EQ(knownWorkerIds.size(), 1);
  ASSERT_TRUE(knownWorkerIds.find(dst_grad) != knownWorkerIds.end());
}

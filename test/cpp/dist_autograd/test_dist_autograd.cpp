#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_context.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/rpc_with_autograd.h>
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

TEST_F(DistAutogradTest, TestSendFunction) {
  // Initialize input tensors requiring grad.
  auto options = at::TensorOptions().requires_grad(true);
  auto in1 = torch::ones({3, 3}, options);
  auto in2 = torch::ones({3, 3}, options);
  ASSERT_FALSE(in1.grad().defined());
  ASSERT_FALSE(in2.grad().defined());

  autogradContainer_->newContext();
  DistAutogradContext& autogradContext = autogradContainer_->currentContext();
  // Attach the send autograd function to tensors.
  std::vector<torch::Tensor> tensors = {in1, in2};
  worker_id_t worker_id = 1;
  addSendRpcBackward(
      autogradContext, AutogradMetadata(1, 1), tensors, worker_id);
  auto send_function = autogradContext.sendFunctions()[1];
  ASSERT_NE(send_function, nullptr);
  // Ensure that worker id is recorded.
  auto knownWorkerIds = autogradContext.getKnownWorkerIds();
  ASSERT_EQ(knownWorkerIds.size(), 1);
  ASSERT_TRUE(knownWorkerIds.find(worker_id) != knownWorkerIds.end());

  // Build loss and attach it as input to send autograd function.
  auto o1 = torch::autograd::Variable(torch::ones({3, 3}));
  auto edge = torch::autograd::Edge(send_function, 0);
  o1.set_gradient_edge(edge);
  auto o2 = torch::autograd::Variable(torch::ones({3, 3}));
  edge = torch::autograd::Edge(send_function, 1);
  o2.set_gradient_edge(edge);
  auto loss = torch::add(o1, o2);

  // Run backwards pass and verify gradients accumulated.
  auto gradient = torch::autograd::Variable(torch::rand({3, 3}));
  loss.backward(gradient, false, false);
  ASSERT_TRUE(in1.grad().defined());
  ASSERT_TRUE(in2.grad().defined());
}

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

  // Build loss and attach it as input to send autograd function.
  auto loss = torch::autograd::Variable(torch::ones({3, 3}));
  loss.set_gradient_edge(torch::autograd::Edge(send_function, 1));

  // This should fail since the SendRpcBackward function is looking for two
  // inputs and as a result encounters an undefined grad.
  EXPECT_THROW(
      loss.backward(torch::autograd::Variable(), false, false), c10::Error);
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

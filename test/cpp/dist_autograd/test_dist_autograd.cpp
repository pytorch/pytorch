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
  addSendRpcBackward(autogradContext, AutogradMetadata(1, 1), tensors);
  auto send_function = autogradContext.sendFunctions()[1];
  ASSERT_NE(send_function, nullptr);

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
  addSendRpcBackward(autogradContext, AutogradMetadata(1, 1), tensors);
  auto send_function = autogradContext.sendFunctions()[1];

  // Build loss and attach it as input to send autograd function.
  auto loss = torch::autograd::Variable(torch::ones({3, 3}));
  loss.set_gradient_edge(torch::autograd::Edge(send_function, 1));

  // This should fail since the SendRpcBackward function is looking for two
  // inputs and as a result encounters an undefined grad.
  EXPECT_THROW(
      loss.backward(torch::autograd::Variable(), false, false), c10::Error);
}

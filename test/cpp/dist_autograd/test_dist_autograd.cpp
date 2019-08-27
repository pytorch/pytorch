#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/torch.h>

TEST(DistAutogradTest, TestSendFunction) {
  // Initialize input tensors requiring grad.
  auto options = at::TensorOptions().requires_grad(true);
  auto in1 = torch::ones({3, 3}, options);
  auto in2 = torch::ones({3, 3}, options);
  ASSERT_FALSE(in1.grad().defined());
  ASSERT_FALSE(in2.grad().defined());

  // Attach the send autograd function to tensors.
  std::vector<at::IValue> ivalues{in1, in2};
  auto send_function =
      torch::distributed::autograd::addSendRpcBackward(ivalues);
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

TEST(DistAutogradTest, TestSendFunctionInvalidInputs) {
  auto options = at::TensorOptions().requires_grad(true);
  auto in1 = torch::ones({3, 3}, options);
  auto in2 = torch::ones({3, 3}, options);

  // Attach the send autograd function to tensors.
  auto send_function =
      torch::distributed::autograd::addSendRpcBackward({in1, in2});

  // Build loss and attach it as input to send autograd function.
  auto loss = torch::autograd::Variable(torch::ones({3, 3}));
  loss.set_gradient_edge(torch::autograd::Edge(send_function, 1));

  // This should fail since the SendRpcBackward function is looking for two
  // inputs and as a result encounters an undefined grad.
  EXPECT_THROW(
      loss.backward(torch::autograd::Variable(), false, false), c10::Error);
}

#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

TEST(InferenceOnlyGuard, ailing) {
  torch::Tensor a = torch::rand({2, 2},torch::requires_grad());
  torch::Tensor b = torch::randn({2, 2});
  torch::Tensor c;
  {
    at::AutoNonVariableTypeMode mode(true);
    c = a + b;
  }
  torch::Tensor d = c + a;
  //std::cout << d.requires_grad() << std::endl;
  //d.backward(torch::ones_like(d));
  //std::cout << a.grad() << std::endl;

  //torch::Tensor e = c.t();
  //torch::Tensor e = c.add_(1);
  //torch::Tensor f = e + a;

}




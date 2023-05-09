#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <torch/torch.h>

extern std::vector<at::Tensor> inductor_entry_cpp(
    const std::vector<at::Tensor>& args);

namespace torch {
namespace aot_inductor {

struct Net : torch::nn::Module {
  Net() : linear(register_module("linear", torch::nn::Linear(64, 10))) {}

  torch::Tensor forward(torch::Tensor x, torch::Tensor y) {
    return linear(torch::sin(x) + torch::cos(y));
  }
  torch::nn::Linear linear;
};

TEST(AotInductorTest, BasicTest) {
  torch::NoGradGuard no_grad;
  Net net;
  net.to(torch::kCUDA);

  torch::Tensor x =
      at::randn({32, 64}, at::dtype(at::kFloat).device(at::kCUDA));
  torch::Tensor y =
      at::randn({32, 64}, at::dtype(at::kFloat).device(at::kCUDA));
  torch::Tensor results_ref = net.forward(x, y);

  // TODO: we need to provide an API to concatenate args and weights
  std::vector<torch::Tensor> inputs;
  for (const auto& pair : net.named_parameters()) {
    inputs.push_back(pair.value());
  }
  inputs.push_back(x);
  inputs.push_back(y);
  auto results_opt = inductor_entry_cpp(inputs);

  ASSERT_TRUE(torch::allclose(results_ref, results_opt[0]));
}

} // namespace aot_inductor
} // namespace torch
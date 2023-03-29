//#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include <torch/torch.h>

extern std::vector<at::Tensor> aot_inductor_entry(const std::vector<at::Tensor>& args);
/*
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.ones(32, 64)

    def forward(self, x):
        x = torch.relu(x + self.weight)
        return x
*/
struct Net : torch::nn::Module {
  Net() {
    weight = register_parameter("weight", torch::ones({32, 64}));
  }
  torch::Tensor forward(torch::Tensor input) {
    return torch::relu(input + weight);
  }
  torch::Tensor weight;
};

int main() {
    torch::Tensor x = at::randn({32, 64});
    Net net;
    torch::Tensor results_ref = net.forward(x);

    // TODO: we need to provide an API to concatenate args and weights
    std::vector<torch::Tensor> inputs;
    for (const auto& pair : net.named_parameters()) {
      inputs.push_back(pair.value());
    }
    inputs.push_back(x);
    auto results_opt = aot_inductor_entry(inputs);

    assert(torch::allclose(results_ref, results_opt[0]));
    printf("PASS\n");
    return 0;
}

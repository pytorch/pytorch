//#include <gtest/gtest.h>
#include <iostream>

#include "build/aot_inductor_output.h"

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
    std::vector<torch::Tensor> inputs = {x};
    for (const auto& pair : net.named_parameters()) {
      inputs.push_back(pair.value());
    }
    torch::Tensor results_opt = aot_inductor_entry(inputs);

    assert(torch::allclose(results_ref, results_opt));
    printf("PASS\n");
    return 0;
}

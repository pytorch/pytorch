#include <torch/torch.h>
#include <iostream>

using namespace torch::nn;


int main() {
  std::cout << "Hello0\n" << std::flush;
  const at::Device device("cuda");
  std::cout << "Hello1\n" << std::flush;
  const auto x = torch::full({1}, 0, torch::TensorOptions().dtype(torch::kUInt8).device(device));
  std::cout << "Hello2\n" << std::flush;
}


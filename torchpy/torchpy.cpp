#include <torchpy.h>
#include <torch/torch.h>
#include <iostream>

void torchpy::init() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}

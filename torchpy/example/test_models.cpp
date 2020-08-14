#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torchpy.h>

TEST(TorchpyTest, SimpleModel) {
  torchpy::init();

  // Load the model
  auto model = torchpy::load("torchpy/example/simple.pt");

  // Execute
  std::vector<at::Tensor> inputs;
  inputs.push_back(torch::ones(at::IntArrayRef({10, 20})));
  auto output = model.forward(inputs);

  std::cout << output << std::endl;

  torchpy::finalize();
}

TEST(TorchpyTest, ResNet) {
  torchpy::init();

  // Load the model
  auto model = torchpy::load("torchpy/example/resnet.pt");

  // Execute
  std::vector<at::Tensor> inputs;
  inputs.push_back(torch::ones(at::IntArrayRef({1, 3, 224, 224})));
  auto output = model.forward(inputs);

  torchpy::finalize();
}

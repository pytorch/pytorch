#include <gtest/gtest.h>
#include <torchpy.h>

TEST(TorchpyTest, ResNet) {
  // Load the model
  auto model = torchpy::load("example/resnet.pt");

  // Generate inputs inside torchpy since torch isn't linked here..
  auto inputs = torchpy::inputs();

  // Execute
  auto output = module.forward(inputs); // .toTensor();
  // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  std::cout << output << std::endl;
}

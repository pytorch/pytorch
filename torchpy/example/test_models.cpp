#include <gtest/gtest.h>
#include <torchpy.h>

TEST(TorchpyTest, SimpleModel) {
  torchpy::init();
  // Load the model
  auto model = torchpy::load("torchpy/example/simple.pt");

  // Generate inputs inside torchpy since torch isn't linked here..

  auto inputs = torchpy::inputs({10, 20});

  // Execut
  auto output = model.forward(inputs); // .toTensor();
  // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  std::cout << output << std::endl;
  torchpy::finalize();
}
TEST(TorchpyTest, ResNet) {
  torchpy::init();
  // Load the model
  auto model = torchpy::load("torchpy/example/resnet.pt");

  // Generate inputs inside torchpy since torch isn't linked here..
  auto inputs = torchpy::inputs({1, 3, 224, 224});

  // Execute
  auto output = model.forward(inputs); // .toTensor();
  // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  // std::cout << output << std::endl;
  torchpy::finalize();
}

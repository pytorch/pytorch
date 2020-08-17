#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchpy.h>

void compare_torchpy_jit(
    const char* model_filename,
    std::vector<at::Tensor> inputs) {
  // Test
  auto model = torchpy::load(model_filename);
  auto output = model.forward(inputs);

  // Reference
  auto ref_model = torch::jit::load(model_filename);
  std::vector<torch::jit::IValue> ref_inputs;
  for (at::Tensor& input : inputs) {
    ref_inputs.push_back(torch::jit::IValue(input));
  }
  auto ref_output = ref_model.forward(ref_inputs).toTensor();

  ASSERT_TRUE(ref_output.equal(output));
}

TEST(TorchpyTest, SimpleModel) {
  torchpy::init();

  compare_torchpy_jit(
      "torchpy/example/simple.pt", {torch::ones(at::IntArrayRef({10, 20}))});

  torchpy::finalize();
}

TEST(TorchpyTest, ResNet) {
  torchpy::init();

  compare_torchpy_jit(
      "torchpy/example/resnet.pt",
      {torch::ones(at::IntArrayRef({1, 3, 224, 224}))});

  torchpy::finalize();
}

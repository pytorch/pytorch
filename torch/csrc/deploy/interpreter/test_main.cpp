#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/deploy/interpreter/interpreter.h>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  int rc = RUN_ALL_TESTS();

  return rc;
}

TEST(Interpreter, Sanity) {
  ASSERT_TRUE(true);
}

TEST(Interpreter, Hello) {
  Interpreter interp;
  interp.run_some_python("print('hello from first interpeter!')");

  Interpreter interp2;
  interp2.run_some_python("print('hello from second interpeter!')");
}

void compare_torchpy_jit(const char* model_filename, at::Tensor const & input) {
  Interpreter interp;
  // Test
  auto model_id = interp.load_model(model_filename, false);
  at::Tensor output = interp.forward_model(model_id, input);

  // Reference
  auto ref_model = torch::jit::load(model_filename);
  std::vector<torch::jit::IValue> ref_inputs;
  ref_inputs.emplace_back(torch::jit::IValue(input));
  at::Tensor ref_output = ref_model.forward(ref_inputs).toTensor();

  ASSERT_TRUE(ref_output.equal(output));
}

TEST(Interpreter, SimpleModel) {
  char* model_path = std::getenv("SIMPLE_MODEL_PATH");
  ASSERT_NE(model_path, nullptr);
  const int A = 10, B = 20;
  compare_torchpy_jit(
      model_path, torch::ones(at::IntArrayRef({A, B})));
}

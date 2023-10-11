#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <torch/csrc/inductor/aoti_model_runner.h>
#include <torch/csrc/inductor/aoti_model_runner_cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

#define STR_VALUE(x) #x
#define STRINGIZE(x) STR_VALUE(x)

namespace torch {
namespace inductor {

TEST(AotInductorTest, BasicTest) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  const auto& model_so_path = data_loader.attr("model_so_path").toStringRef();
  const auto& input_tensors = data_loader.attr("inputs").toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr("outputs").toTensorList().vec();

  AOTIModelRunnerCuda runner(model_so_path.c_str());
  auto actual_output_tensors = runner.run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}

} // namespace inductor
} // namespace torch

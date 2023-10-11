#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <vector>

#include <torch/csrc/inductor/aoti_model_runner.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_model_runner_cuda.h>
#endif
#include <torch/script.h>
#include <torch/torch.h>

#define STR_VALUE(x) #x
#define STRINGIZE(x) STR_VALUE(x)

namespace torch {
namespace inductor {

TEST(AotInductorTest, BasicTestCpu) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  const auto& model_so_path =
      data_loader.attr("model_so_path_cpu").toStringRef();
  const auto& input_tensors =
      data_loader.attr("inputs_cpu").toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr("outputs_cpu").toTensorList().vec();

  AOTIModelRunnerCpu runner(model_so_path.c_str());
  auto actual_output_tensors = runner.run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}

#ifdef USE_CUDA
TEST(AotInductorTest, BasicTestCuda) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  const auto& model_so_path =
      data_loader.attr("model_so_path_cuda").toStringRef();
  const auto& input_tensors =
      data_loader.attr("inputs_cuda").toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr("outputs_cuda").toTensorList().vec();

  AOTIModelRunnerCuda runner(model_so_path.c_str());
  auto actual_output_tensors = runner.run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}
#endif

} // namespace inductor
} // namespace torch

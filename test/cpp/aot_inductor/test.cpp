#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <vector>

#include <torch/csrc/inductor/aoti_model_container_runner.h>
#include <torch/csrc/inductor/aoti_model_runner.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_model_container_runner_cuda.h>
#endif
#include <torch/script.h>
#include <torch/torch.h>

#define STR_VALUE(x) #x
#define STRINGIZE(x) STR_VALUE(x)

namespace torch {
namespace inductor {

TEST(AotInductorModelTest, BasicTestCpu) {
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

  const auto& weight_tensors = data_loader.attr("fc_weight_cpu").toTensor();
  const auto& bias_tensors = data_loader.attr("fc_bias_cpu").toTensor();

  ConstantMap const_map;
  const_map.emplace("fc_weight", new at::Tensor(weight_tensors));
  const_map.emplace("fc_bias", new at::Tensor(bias_tensors));

  AOTIModelRunnerCpu runner(model_so_path.c_str(), const_map);
  auto actual_output_tensors = runner.run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}

TEST(AotInductorModelTest, UpdateConstantsTestCpu) {
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

  const auto& weight_tensors = data_loader.attr("fc_weight_cpu").toTensor();
  const auto& bias_tensors = data_loader.attr("fc_bias_cpu").toTensor();

  ConstantMap rand_map, w_map, b_map;
  rand_map.emplace("fc_weight", new at::Tensor(at::randn({10, 64})));
  rand_map.emplace("fc_bias", new at::Tensor(at::randn({10})));
  w_map.emplace("fc_weight", new at::Tensor(weight_tensors));
  b_map.emplace("fc_bias", new at::Tensor(bias_tensors));

  AOTIModelRunnerCpu runner(model_so_path.c_str(), rand_map);
  auto actual_output_tensors = runner.run(input_tensors);
  ASSERT_FALSE(
      torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  runner.update_constants(w_map);
  actual_output_tensors = runner.run(input_tensors);
  ASSERT_FALSE(
      torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  runner.update_constants(b_map);
  actual_output_tensors = runner.run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}

TEST(AotInductorModelTest, UpdateConstantsMapTestCpu) {
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

  const auto& weight_tensors = data_loader.attr("fc_weight_cpu").toTensor();
  const auto& bias_tensors = data_loader.attr("fc_bias_cpu").toTensor();

  ConstantMap rand_map, real_map;
  rand_map.emplace("fc_weight", new at::Tensor(at::randn({10, 64})));
  rand_map.emplace("fc_bias", new at::Tensor(at::randn({10})));
  real_map.emplace("fc_weight", new at::Tensor(weight_tensors));
  real_map.emplace("fc_bias", new at::Tensor(bias_tensors));

  AOTIModelRunnerCpu runner(model_so_path.c_str(), rand_map);
  auto actual_output_tensors = runner.run(input_tensors);
  ASSERT_FALSE(
      torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  runner.update_constants_map(real_map);
  actual_output_tensors = runner.run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}

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

  AOTIModelContainerRunnerCpu runner(model_so_path.c_str());
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

  AOTIModelContainerRunnerCuda runner(model_so_path.c_str());
  auto actual_output_tensors = runner.run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}
#endif

} // namespace inductor
} // namespace torch

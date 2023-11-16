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

namespace {

template <typename RunnerT>
void test_aoti(const std::string& device) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string path_attr = "model_so_path_" + device;
  std::string inputs_attr = "inputs_" + device;
  std::string outputs_attr = "outputs_" + device;
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  const auto& input_tensors =
      data_loader.attr(inputs_attr.c_str()).toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  auto runner = std::make_unique<RunnerT>(model_so_path.c_str());

  auto actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}

void test_aoti_script(const std::string& device) {
  torch::NoGradGuard no_grad;

  std::string script_model = "script_model_" + device + ".pt";
  std::string model_path =
      (std::filesystem::path(
           STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / script_model.c_str())
           .string();
  torch::jit::script::Module model = torch::jit::load(model_path);

  std::string sample_data_path =
      (std::filesystem::path(
           STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "script_data.pt")
           .string();
  torch::jit::script::Module sample_data = torch::jit::load(sample_data_path);
  std::string inputs_attr = "inputs_" + device;
  std::string outputs_attr = "outputs_" + device;
  const auto& inputs = sample_data.attr(inputs_attr.c_str()).toList().vec();
  const auto& ref_output_tensors =
      sample_data.attr(outputs_attr.c_str()).toTensorVector();
  auto outputs = model.forward(inputs).toTuple()->elements();
  ASSERT_EQ(outputs.size(), ref_output_tensors.size());
  for (size_t i = 0; i < ref_output_tensors.size(); i++) {
    ASSERT_TRUE(torch::allclose(outputs[i].toTensor(), ref_output_tensors[i]));
  }
}

} // namespace

namespace torch {
namespace inductor {

TEST(AotInductorTest, BasicModelTestCpu) {
  test_aoti<torch::inductor::AOTIModelRunnerCpu>("cpu");
}

TEST(AotInductorTest, BasicModelContainerTestCpu) {
  test_aoti<torch::inductor::AOTIModelContainerRunnerCpu>("cpu");
}

TEST(AotInductorTest, BasicScriptTestCpu) {
  test_aoti_script("cpu");
}

#ifdef USE_CUDA
TEST(AotInductorTest, BasicModelContainerTestCuda) {
  test_aoti<torch::inductor::AOTIModelContainerRunnerCuda>("cuda");
}

TEST(AotInductorTest, BasicScriptTestCuda) {
  test_aoti_script("cuda");
}
#endif

} // namespace inductor
} // namespace torch

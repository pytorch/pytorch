#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchpy.h>
#include <unistd.h>
#include <future>

TEST(TorchpyTest, DISABLED_MultiSerialSimpleModel) {
  auto model_filename = "torchpy/example/simple.pt";
  auto input = torch::ones(at::IntArrayRef({10, 20}));
  size_t ninterp = 3;
  std::vector<at::Tensor> outputs;

  // Shared model
  auto model_id = torchpy::load(model_filename);

  // Futures on model forward
  for (size_t i = 0; i < ninterp; i++) {
    auto output = torchpy::forward(model_id, input);
    outputs.push_back(output);
  }

  // Generate reference
  auto ref_model = torch::jit::load(model_filename);
  std::vector<torch::jit::IValue> ref_inputs;
  ref_inputs.push_back(torch::jit::IValue(input));
  auto ref_output = ref_model.forward(ref_inputs).toTensor();

  // Compare all to reference
  for (size_t i = 0; i < ninterp; i++) {
    ASSERT_TRUE(ref_output.equal(outputs[i]));
  }
}

TEST(TorchpyTest, ThreadedSimpleModel) {
  auto model_filename = "torchpy/example/simple.pt";
  size_t nthreads = 2;
  std::vector<at::Tensor> outputs;
  auto input = torch::ones(at::IntArrayRef({10, 20}));

  // Shared model
  auto model_id = torchpy::load(model_filename);

  // Futures on model forward
  std::vector<std::future<at::Tensor>> futures;
  for (size_t i = 0; i < nthreads; i++) {
    futures.push_back(std::async(std::launch::async, [model_id]() {
      auto input = torch::ones(at::IntArrayRef({10, 20}));
      auto result = torchpy::forward(model_id, input);
      return result;
    }));
  }
  std::cout << "pushed all futures" << std::endl;
  usleep(100 * 1000);
  for (size_t i = 0; i < nthreads; i++) {
    outputs.push_back(futures[i].get());
  }
  std::cout << "got all outputs" << std::endl;

  // Generate reference
  auto ref_model = torch::jit::load(model_filename);
  std::vector<torch::jit::IValue> ref_inputs;
  ref_inputs.push_back(torch::jit::IValue(input));
  auto ref_output = ref_model.forward(ref_inputs).toTensor();
  std::cout << "got ref output" << std::endl;

  // Compare all to reference
  for (size_t i = 0; i < nthreads; i++) {
    ASSERT_TRUE(ref_output.equal(outputs[i]));
  }
  std::cout << "Passed assertions" << std::endl;
}

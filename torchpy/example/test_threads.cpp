#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchpy.h>
#include <future>

TEST(TorchpyTest, ThreadedSimpleModel) {
  auto model_filename = "torchpy/example/simple.pt";
  auto input = torch::ones(at::IntArrayRef({10, 20}));
  size_t nthreads = 2;
  std::vector<at::Tensor> outputs;

  // Shared model
  auto model = torchpy::load(model_filename);

  // Futures on model forward
  std::vector<std::future<at::Tensor>> futures;
  for (size_t i = 0; i < nthreads; i++) {
    futures.push_back(std::async(std::launch::async, [&model, &input]() {
      model.thread_begin();
      auto result = model.forward(input);
      model.thread_end();
      return result;
    }));
  }

  for (size_t i = 0; i < nthreads; i++) {
    outputs.push_back(futures[i].get());
  }

  // Generate reference
  auto ref_model = torch::jit::load(model_filename);
  std::vector<torch::jit::IValue> ref_inputs;
  ref_inputs.push_back(torch::jit::IValue(input));
  auto ref_output = ref_model.forward(ref_inputs).toTensor();

  // Compare all to reference
  for (size_t i = 0; i < nthreads; i++) {
    ASSERT_TRUE(ref_output.equal(outputs[i]));
  }
}

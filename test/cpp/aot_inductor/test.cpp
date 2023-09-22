#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aot_runtime/interface.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/torch.h>

namespace torch {
namespace aot_inductor {

struct Net : torch::nn::Module {
  Net() : linear(register_module("linear", torch::nn::Linear(64, 10))) {}

  torch::Tensor forward(torch::Tensor x, torch::Tensor y) {
    return linear(torch::sin(x) + torch::cos(y));
  }
  torch::nn::Linear linear;
};

TEST(AotInductorTest, BasicTest) {
  torch::NoGradGuard no_grad;
  Net net;
  net.to(torch::kCUDA);

  // We should fix the weight over here.
  // This should match exactly with the one in test.py
  torch::Tensor weights =
      at::arange(640, at::dtype(at::kFloat).device(at::kCUDA));
  weights = at::reshape(weights, {10, 64});
  torch::Tensor bias = at::zeros({10}, at::dtype(at::kFloat).device(at::kCUDA));

  for (const auto& pair : net.named_parameters()) {
    if (pair.key().find("weight") != std::string::npos) {
      pair.value().copy_(weights);
    } else if (pair.key().find("bias") != std::string::npos) {
      pair.value().copy_(bias);
    }
  }

  torch::Tensor x =
      at::randn({32, 64}, at::dtype(at::kFloat).device(at::kCUDA));
  torch::Tensor y =
      at::randn({32, 64}, at::dtype(at::kFloat).device(at::kCUDA));
  torch::Tensor results_ref = net.forward(x, y);
  std::vector<torch::Tensor> input_tensors;
  input_tensors.push_back(x);
  input_tensors.push_back(y);

  AOTInductorModelContainerHandle container_handle;
  AOTI_RUNTIME_ERROR_CODE_CHECK(AOTInductorModelContainerCreate(
      &container_handle,
      1 /*num_models*/,
      false /*is_cpu*/,
      nullptr /*cubin_dir*/));

  auto input_handles =
      torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(input_tensors);

  // For outputs, we only allocate a vector to hold returned tensor handles,
  // not allocating the actual output tensor storage here
  size_t num_outputs;
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      AOTInductorModelContainerGetNumOutputs(container_handle, &num_outputs));
  std::vector<AtenTensorHandle> output_handles(num_outputs);

  const auto& cuda_stream = at::cuda::getCurrentCUDAStream(0 /*device_index*/);
  const auto stream_id = cuda_stream.stream();
  AOTInductorStreamHandle stream_handle =
      reinterpret_cast<AOTInductorStreamHandle>(stream_id);

  AOTIProxyExecutorHandle proxy_executor_handle = nullptr;

  AOTI_RUNTIME_ERROR_CODE_CHECK(AOTInductorModelContainerRun(
      container_handle,
      input_handles.data(),
      input_tensors.size(),
      output_handles.data(),
      output_handles.size(),
      stream_handle,
      proxy_executor_handle));

  auto outputs = torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
      output_handles.data(), output_handles.size());

  ASSERT_TRUE(torch::allclose(results_ref, outputs[0]));
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      AOTInductorModelContainerDelete(container_handle));
}

} // namespace aot_inductor
} // namespace torch

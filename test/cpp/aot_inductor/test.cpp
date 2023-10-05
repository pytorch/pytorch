#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/script.h>
#include <torch/torch.h>

#define STR_VALUE(x) #x
#define STRINGIZE(x) STR_VALUE(x)

namespace torch {
namespace aot_inductor {

class RAIIModelContainer {
 public:
  RAIIModelContainer() {
    AOTI_RUNTIME_ERROR_CODE_CHECK(AOTInductorModelContainerCreate(
        &container_handle,
        1 /*num_models*/,
        false /*is_cpu*/,
        nullptr /*cubin_dir*/));
  }

  ~RAIIModelContainer() {
    AOTI_RUNTIME_ERROR_CODE_CHECK(
        AOTInductorModelContainerDelete(container_handle));
  }

  AOTInductorModelContainerHandle get() const {
    return container_handle;
  }

 private:
  AOTInductorModelContainerHandle container_handle;
};

TEST(AotInductorTest, BasicTest) {
  torch::NoGradGuard no_grad;

  std::filesystem::path io_tensors_path = std::filesystem::path(
      STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "io_tensors.pt";
  torch::jit::script::Module tensor_loader =
      torch::jit::load(io_tensors_path.string());
  torch::Tensor x = tensor_loader.attr("x").toTensor();
  torch::Tensor y = tensor_loader.attr("y").toTensor();
  std::vector<torch::Tensor> input_tensors = {x, y};
  torch::Tensor ref_output = tensor_loader.attr("output").toTensor();

  RAIIModelContainer model_container;
  // TODO: will add another API to avoid doing all these input preparations
  auto input_handles =
      torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(input_tensors);

  // For outputs, we only allocate a vector to hold returned tensor handles,
  // not allocating the actual output tensor storage here
  size_t num_outputs;
  AOTI_RUNTIME_ERROR_CODE_CHECK(AOTInductorModelContainerGetNumOutputs(
      model_container.get(), &num_outputs));
  std::vector<AtenTensorHandle> output_handles(num_outputs);

  const auto& cuda_stream = at::cuda::getCurrentCUDAStream(0 /*device_index*/);
  const auto stream_id = cuda_stream.stream();
  AOTInductorStreamHandle stream_handle =
      reinterpret_cast<AOTInductorStreamHandle>(stream_id);

  AOTIProxyExecutorHandle proxy_executor_handle = nullptr;

  AOTI_RUNTIME_ERROR_CODE_CHECK(AOTInductorModelContainerRun(
      model_container.get(),
      input_handles.data(),
      input_tensors.size(),
      output_handles.data(),
      output_handles.size(),
      stream_handle,
      proxy_executor_handle));

  auto actual_output_tensors =
      torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
          output_handles.data(), output_handles.size());

  ASSERT_TRUE(torch::allclose(ref_output, actual_output_tensors[0]));
}

} // namespace aot_inductor
} // namespace torch

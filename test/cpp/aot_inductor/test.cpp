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

TEST(AotInductorTest, BasicTest) {
  torch::NoGradGuard no_grad;

  std::string io_tensors_path =
      (std::filesystem::path(
           STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "io_tensors.pt")
           .string();
  torch::jit::script::Module tensor_loader = torch::jit::load(io_tensors_path);
  auto input_tensors = tensor_loader.attr("inputs").toTensorList().vec();
  auto ref_output_tensors = tensor_loader.attr("outputs").toTensorList().vec();

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

  auto output_tensors =
      torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
          output_handles.data(), output_handles.size());

  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], output_tensors[0]));
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      AOTInductorModelContainerDelete(container_handle));
}

} // namespace aot_inductor
} // namespace torch

#pragma once

#include <vector>

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>

// Define AOTI_STANDALONE here to avoid redefined symbols seen by linter
#ifndef AOTI_STANDALONE
#define AOTI_STANDALONE
#endif
#include <torch/csrc/inductor/aoti_runtime/model_container.h>

#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_standalone/cuda/utils.h>
#endif

namespace torch::standalone {
namespace {
std::vector<SlimTensor*> unsafe_alloc_new_handles_from_tensors(
    const std::vector<SlimTensor>& tensors) {
  std::vector<SlimTensor*> result;
  result.reserve(tensors.size());
  for (auto tensor : tensors) {
    auto allocated = new SlimTensor(std::move(tensor));
    result.push_back(allocated);
  }
  return result;
}

std::vector<SlimTensor> alloc_tensors_by_stealing_from_handles(
    std::vector<SlimTensor*>& outputs) {
  // Find duplicates by recording the last known index for each handle.
  std::unordered_map<SlimTensor*, size_t> lastKnownIdx;
  size_t length = outputs.size();
  for (size_t i = 0; i < length; i++) {
    lastKnownIdx[outputs[i]] = i;
  }

  std::vector<SlimTensor> result;
  result.reserve(length);
  for (size_t i = 0; i < length; i++) {
    if (outputs[i] == nullptr) {
      result.emplace_back();
      continue;
    }

    SlimTensor tensor = *outputs[i];
    result.emplace_back(std::move(tensor));
    if (lastKnownIdx[outputs[i]] == i) {
      delete outputs[i];
    }
  }
  outputs.clear();
  return result;
}
} // namespace

class SlimTensorRunner : public torch::aot_inductor::AOTInductorModelContainer {
 public:
  SlimTensorRunner(const c10::Device& device)
      : torch::aot_inductor::AOTInductorModelContainer(1, device.str()) {}

  std::vector<SlimTensor> run(const std::vector<SlimTensor>& inputs) {
    std::vector<SlimTensor*> input_handles =
        unsafe_alloc_new_handles_from_tensors(inputs);
    std::vector<SlimTensor*> output_handles(this->num_outputs());

#ifdef USE_CUDA
    AOTICudaStream cuda_stream;
    torch::aot_inductor::DeviceStreamType stream = cuda_stream.get();
#else
    torch::aot_inductor::DeviceStreamType stream = nullptr;
#endif
    // auto* model = static_cast<Model*>(this);
    this->run_single_threaded(
        input_handles.data(), output_handles.data(), stream, nullptr);
    return alloc_tensors_by_stealing_from_handles(output_handles);
  }
};
} // namespace torch::standalone

#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include <cuda_runtime.h>

namespace c10d {

std::vector<at::Tensor> getTensorShapes(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> shapeTensors;
  shapeTensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    // Use `at::tensor()` to copy the data underlying `sizes()` since it may be
    // released elsewhere.
    at::Tensor shapesTensor =
        at::tensor(tensor.sizes(), at::TensorOptions().dtype(at::kLong));
    shapeTensors.emplace_back(std::move(shapesTensor));
  }
  return shapeTensors;
}

size_t hashTensors(const std::vector<at::Tensor>& tensors) {
  size_t hash = 0;
  for (auto& tensor : tensors) {
    size_t data_size = tensor.storage().nbytes();
    std::hash<char> hasher;
    auto src = static_cast<const char*>(tensor.storage().data_ptr().get());
    char* dst = (char*)std::calloc(data_size, sizeof(char));
    cudaMemcpy(dst, src, data_size, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < data_size; ++i) {
      // Update the hash for each byte in the tensor
      hash ^= hasher(((char*)dst)[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    free(dst);
  }
  return hash;
}

size_t getTensorsNumel(const std::vector<at::Tensor>& tensors) {
  size_t numel = 0;
  for (auto& tensor : tensors) {
    numel += tensor.numel();
  }
  return numel;
}

} // namespace c10d

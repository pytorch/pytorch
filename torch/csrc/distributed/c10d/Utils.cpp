#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

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

size_t getTensorsNumel(const std::vector<at::Tensor>& tensors) {
  size_t numel = 0;
  for (auto& tensor : tensors) {
    numel += tensor.numel();
  }
  return numel;
}

} // namespace c10d

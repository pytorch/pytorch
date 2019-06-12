#include <torch/optim/serialize.h>

#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace torch {
namespace optim {
namespace detail {
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const std::vector<int64_t>& steps) {
  std::vector<torch::Tensor> tensors;
  for (const auto& step : steps) {
    tensors.push_back(torch::tensor(static_cast<int64_t>(step)));
  }
  serialize(archive, key, tensors);
}

void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    std::vector<int64_t>& steps) {
  std::vector<torch::Tensor> tensors;
  serialize(archive, key, tensors);
  steps.clear();
  for (const auto& step : tensors) {
    steps.push_back(step.item<int64_t>());
  }
}
} // namespace detail
} // namespace optim
} // namespace torch

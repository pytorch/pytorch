#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <cstring>

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

void getGlobalRankStartAndStride(
    const std::vector<uint64_t>& globalRanksInGroup,
    int& globalRankStart,
    int& globalRankStride) {
  if (globalRanksInGroup.empty()) {
    globalRankStart = 0;
  } else {
    globalRankStart = static_cast<int>(globalRanksInGroup[0]);
  }

  if (globalRanksInGroup.empty()) {
    globalRankStride = 1;
  } else if (globalRanksInGroup.size() == 1) {
    globalRankStride = 0;
  } else {
    bool ranksAreStrided = true;
    auto startRank = globalRanksInGroup[0];
    auto stride = globalRanksInGroup[1] - globalRanksInGroup[0];
    for (std::vector<uint64_t>::size_type i = 0; i < globalRanksInGroup.size();
         i++) {
      if (globalRanksInGroup[i] != startRank + i * stride) {
        ranksAreStrided = false;
        break;
      }
    }

    if (ranksAreStrided) {
      globalRankStride =
          static_cast<int>(globalRanksInGroup[1] - globalRanksInGroup[0]);
    } else {
      globalRankStride = -1;
    }
  }
}

} // namespace c10d

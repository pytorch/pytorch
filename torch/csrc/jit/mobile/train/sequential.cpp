#include <torch/csrc/jit/mobile/train/sequential.h>
#include <torch/types.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace torch::jit::mobile {
SequentialSampler::SequentialSampler(size_t size) : size_(size) {}

void SequentialSampler::reset(std::optional<size_t> new_size) {
  if (new_size.has_value()) {
    size_ = *new_size;
  }
  index_ = 0;
}

optional<std::vector<size_t>> SequentialSampler::next(size_t batch_size) {
  const auto remaining_indices = size_ - index_;
  if (remaining_indices == 0) {
    return nullopt;
  }
  std::vector<size_t> index_batch(std::min(batch_size, remaining_indices));
  for (auto& i : index_batch) {
    i = index_++;
  }
  return index_batch;
}

void SequentialSampler::save(serialize::OutputArchive& archive) const {
  TORCH_CHECK(
      false, "Serialization of SequentialSampler not supported on mobile.");
}

void SequentialSampler::load(serialize::InputArchive& archive) {
  TORCH_CHECK(
      false, "Serialization of SequentialSampler not supported on mobile.");
}

size_t SequentialSampler::index() const noexcept {
  return index_;
}

} // namespace torch::jit::mobile

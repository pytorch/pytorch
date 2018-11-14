#include <torch/data/samplers/sequential.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace samplers {
SequentialSampler::SequentialSampler(size_t size) : size_(size) {}

void SequentialSampler::reset() {
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
  archive.write(
      "index",
      torch::tensor(static_cast<int64_t>(index_), torch::kInt64),
      /*is_buffer=*/true);
}

void SequentialSampler::load(serialize::InputArchive& archive) {
  auto tensor = torch::empty(1, torch::kInt64);
  archive.read(
      "index",
      tensor,
      /*is_buffer=*/true);
  index_ = tensor.item<int64_t>();
}

size_t SequentialSampler::index() const noexcept {
  return index_;
}

} // namespace samplers
} // namespace data
} // namespace torch

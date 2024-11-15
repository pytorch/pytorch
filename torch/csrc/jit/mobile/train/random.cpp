#include <torch/csrc/jit/mobile/train/random.h>
#include <torch/types.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace torch::jit::mobile {

RandomSampler::RandomSampler(int64_t size, Dtype index_dtype)
    : indices_(torch::randperm(size, index_dtype)) {}

RandomSampler::~RandomSampler() = default;

void RandomSampler::reset(std::optional<size_t> new_size) {
  // This allocates a new chunk of memory every time (just FYI). It should be
  // amortized over the entire epoch hopefully.
  const auto size = new_size.value_or(static_cast<size_t>(indices_.numel()));
  indices_ = torch::randperm(static_cast<int64_t>(size), indices_.options());
  index_ = 0;
}

std::optional<std::vector<size_t>> RandomSampler::next(size_t batch_size) {
  AT_ASSERT(index_ <= indices_.numel());
  const size_t remaining_indices = indices_.numel() - index_;
  if (remaining_indices == 0) {
    return std::nullopt;
  }
  std::vector<size_t> index_batch(std::min(batch_size, remaining_indices));
  auto slice = indices_.slice(/*dim=*/0, index_, index_ + index_batch.size());
  // You may want to store your indices with 32-bit or less, but here we need
  // to upcast to 64-bit. A batch itself won't hold too many indices, so that
  // should be ok. Note that if this indeed results in a type promotion, there
  // will be two allocations: one for the upcast slice, and one for the
  // returned `index_batch` vector.
  slice = slice.to(torch::kInt64);
  const auto* data = slice.const_data_ptr<int64_t>();
  std::copy(data, data + index_batch.size(), index_batch.begin());
  index_ += static_cast<int64_t>(index_batch.size());
  return index_batch;
}

void RandomSampler::save(serialize::OutputArchive& archive) const {
  TORCH_CHECK(false, "Serialization of RandomSampler not supported on mobile.");
}

void RandomSampler::load(serialize::InputArchive& archive) {
  TORCH_CHECK(false, "Serialization of RandomSampler not supported on mobile.");
}

size_t RandomSampler::index() const noexcept {
  return index_;
}

} // namespace torch::jit::mobile

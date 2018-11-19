#include <torch/data/samplers/stream.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>

namespace torch {
namespace data {
namespace samplers {

BatchSize::BatchSize(size_t size) : size_(size) {}
size_t BatchSize::size() const noexcept {
  return size_;
}
BatchSize::operator size_t() const noexcept {
  return size_;
}

StreamSampler::StreamSampler(size_t epoch_size) : epoch_size_(epoch_size) {}

void StreamSampler::reset() {
  examples_retrieved_so_far_ = 0;
}

optional<BatchSize> StreamSampler::next(size_t batch_size) {
  AT_ASSERT(examples_retrieved_so_far_ <= epoch_size_);
  if (examples_retrieved_so_far_ == epoch_size_) {
    return nullopt;
  }
  if (examples_retrieved_so_far_ + batch_size > epoch_size_) {
    batch_size = epoch_size_ - examples_retrieved_so_far_;
  }
  examples_retrieved_so_far_ += batch_size;
  return BatchSize(batch_size);
}

void StreamSampler::save(serialize::OutputArchive& archive) const {
  archive.write(
      "examples_retrieved_so_far",
      torch::tensor(
          static_cast<int64_t>(examples_retrieved_so_far_), torch::kInt64),
      /*is_buffer=*/true);
}

void StreamSampler::load(serialize::InputArchive& archive) {
  auto tensor = torch::empty(1, torch::kInt64);
  archive.read(
      "examples_retrieved_so_far",
      tensor,
      /*is_buffer=*/true);
  examples_retrieved_so_far_ = tensor.item<int64_t>();
}

} // namespace samplers
} // namespace data
} // namespace torch

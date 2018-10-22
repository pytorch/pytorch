#include <torch/data/samplers/stream.h>
#include <torch/tensor.h>

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
  index_ = 0;
}

optional<BatchSize> StreamSampler::next(size_t batch_size) {
  AT_ASSERT(index_ <= epoch_size_);
  if (index_ == epoch_size_) {
    return nullopt;
  }
  if (index_ + batch_size > epoch_size_) {
    batch_size = epoch_size_ - index_;
  }
  index_ += batch_size;
  return BatchSize(batch_size);
}
} // namespace samplers
} // namespace data
} // namespace torch

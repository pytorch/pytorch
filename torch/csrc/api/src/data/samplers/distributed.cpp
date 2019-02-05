#include <torch/data/samplers/distributed.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

namespace torch {
namespace data {
namespace samplers {

DistributedSampler::DistributedSampler(
    size_t size,
    size_t num_replicas,
    size_t rank,
    bool allow_duplicates)
    : size_(size), num_replicas_(num_replicas), rank_(rank), epoch_(0) {
  if (allow_duplicates) {
    local_sample_count_ =
        static_cast<size_t>(std::ceil(size_ * 1.0 / num_replicas_));
  } else {
    local_sample_count_ =
        static_cast<size_t>(std::floor(size_ * 1.0 / num_replicas_));
  }
}

void DistributedSampler::set_epoch(size_t epoch) {
  epoch_ = epoch;
}

size_t DistributedSampler::local_sample_count() {
  return local_sample_count_;
}

DistributedRandomSampler::DistributedRandomSampler(
    size_t size,
    size_t num_replicas,
    size_t rank,
    bool allow_duplicates)
    : DistributedSampler(size, num_replicas, rank, allow_duplicates) {
  populate_indices();
  // shuffle first time.
  reset(size_);
}

optional<std::vector<size_t>> DistributedRandomSampler::next(
    size_t batch_size) {
  if (sample_index_ == end_index_) {
    return nullopt;
  }

  size_t end = sample_index_ + batch_size;
  if (end > end_index_) {
    end = end_index_;
  }

  auto iter = all_indices_.begin();
  std::vector<size_t> res(iter + sample_index_, iter + end);
  sample_index_ = end;
  return res;
}

void DistributedRandomSampler::reset(optional<size_t> new_size) {
  size_t size = new_size.value_or(size_);
  if (size != size_) {
    size_ = size;
    populate_indices();
  }
  std::minstd_rand rand(epoch_);
  std::shuffle(all_indices_.begin(), all_indices_.end(), rand);
  sample_index_ = begin_index_;
}

void DistributedRandomSampler::populate_indices() {
  size_t sample_count =
      num_replicas_ == 1 ? size_ : local_sample_count_ * num_replicas_;
  all_indices_.resize(sample_count);
  std::iota(std::begin(all_indices_), std::end(all_indices_), 0);
  if (num_replicas_ > 1 && sample_count > size_) {
    for (size_t i = size_; i < sample_count; ++i) {
      all_indices_[i] =
          i % size_; // we may have added duplicate samples to make all
                     // replicas to have the same number of samples.
    }
  }

  begin_index_ = rank_ * local_sample_count_;
  end_index_ = begin_index_ + local_sample_count_;
  sample_index_ = begin_index_;
}

void DistributedRandomSampler::save(serialize::OutputArchive& archive) const {
  archive.write(
      "sample_index_",
      torch::tensor(static_cast<int64_t>(sample_index_), torch::kInt64),
      /*is_buffer=*/true);
}

void DistributedRandomSampler::load(serialize::InputArchive& archive) {
  auto tensor = torch::empty(1, torch::kInt64);
  archive.read(
      "sample_index_",
      tensor,
      /*is_buffer=*/true);
  sample_index_ = tensor.item<int64_t>();
}

size_t DistributedRandomSampler::index() const noexcept {
  return sample_index_;
}

DistributedSequentialSampler::DistributedSequentialSampler(
    size_t size,
    size_t num_replicas,
    size_t rank,
    bool allow_duplicates)
    : DistributedSampler(size, num_replicas, rank, allow_duplicates) {
  populate_indices();
}

optional<std::vector<size_t>> DistributedSequentialSampler::next(
    size_t batch_size) {
  if (sample_index_ == end_index_) {
    return nullopt;
  }

  size_t end = sample_index_ + batch_size;
  if (end > end_index_) {
    end = end_index_;
  }

  std::vector<size_t> res(end - sample_index_);
  std::iota(std::begin(res), std::end(res), sample_index_);
  if (end >= size_) {
    for (size_t i = 0; i < res.size(); ++i) {
      res[i] = res[i] % size_;
    }
  }
  sample_index_ = end;
  return res;
}

void DistributedSequentialSampler::reset(optional<size_t> new_size) {
  size_t size = new_size.value_or(size_);
  if (size != size_) {
    size_ = size;
    populate_indices();
  } else {
    sample_index_ = begin_index_;
  }
}

void DistributedSequentialSampler::populate_indices() {
  begin_index_ = rank_ * local_sample_count_;
  end_index_ = begin_index_ + local_sample_count_;
  sample_index_ = begin_index_;
}

void DistributedSequentialSampler::save(
    serialize::OutputArchive& archive) const {
  archive.write(
      "sample_index_",
      torch::tensor(static_cast<int64_t>(sample_index_), torch::kInt64),
      /*is_buffer=*/true);
}

void DistributedSequentialSampler::load(serialize::InputArchive& archive) {
  auto tensor = torch::empty(1, torch::kInt64);
  archive.read(
      "sample_index_",
      tensor,
      /*is_buffer=*/true);
  sample_index_ = tensor.item<int64_t>();
}

size_t DistributedSequentialSampler::index() const noexcept {
  return sample_index_;
}

} // namespace samplers
} // namespace data
} // namespace torch
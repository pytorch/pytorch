#include <ATen/ThreadLocalState.h>
#include <torch/csrc/distributed/c10d/sequence_num.hpp>

#include <c10/util/Logging.h>

namespace c10d {
SequenceNum::SequenceNum() = default;

SequenceNum::SequenceNum(const uint64_t num) : num_(num) {}

SequenceNum::SequenceNum(const SequenceNum& other) {
  if (!other.isSet()) {
    num_ = std::nullopt;
  } else {
    num_ = other.get();
  }
}

uint64_t SequenceNum::get() const {
  std::lock_guard<std::mutex> lock(lock_);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return num_.value();
}

void SequenceNum::increment() {
  std::lock_guard<std::mutex> lock(lock_);
  TORCH_CHECK(num_.has_value());
  num_ = ++(*num_);
}

// Implemented without above get() and increment() so we don't repeatedly lock
// and unblock.
uint64_t SequenceNum::getAndIncrement() {
  uint64_t curVal = 0;
  std::lock_guard<std::mutex> lock(lock_);
  TORCH_CHECK(num_.has_value());
  curVal = *num_;
  num_ = ++(*num_);
  return curVal;
}

void SequenceNum::set(const uint64_t num) {
  std::lock_guard<std::mutex> lock(lock_);
  num_ = num;
}

bool SequenceNum::isSet() const {
  std::lock_guard<std::mutex> lock(lock_);
  return num_.has_value();
}

SequenceNum& SequenceNum::operator=(const SequenceNum& other) {
  std::lock_guard<std::mutex> lock(lock_);
  if (!other.isSet()) {
    num_ = std::nullopt;
  } else {
    num_ = other.get();
  }
  return *this;
}

} // namespace c10d

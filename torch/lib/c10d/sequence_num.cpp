#include <ATen/ThreadLocalState.h>
#include <c10d/sequence_num.hpp>

#include <c10/util/Logging.h>

namespace c10d {
SequenceNum::SequenceNum() : num_(kUnsetSeqNum), set_(false) {}

SequenceNum::SequenceNum(const uint64_t num) : num_(num), set_(true) {}

SequenceNum::SequenceNum(const SequenceNum& other) {
  if (!other.isSet()) {
    num_ = 0;
    set_ = false;
  } else {
    num_ = other.get();
    set_ = true;
  }
}

uint64_t SequenceNum::get() const {
  std::lock_guard<std::mutex> lock(lock_);
  TORCH_CHECK(set_);
  return num_;
}

void SequenceNum::increment() {
  std::lock_guard<std::mutex> lock(lock_);
  TORCH_CHECK(set_);
  ++num_;
}

// Implemented without above get() and increment() so we don't repeatedly lock
// and unblock.
uint64_t SequenceNum::getAndIncrement() {
  uint64_t curVal;
  std::lock_guard<std::mutex> lock(lock_);
  TORCH_CHECK(set_);
  curVal = num_++;
  return curVal;
}

void SequenceNum::set(const uint64_t num) {
  std::lock_guard<std::mutex> lock(lock_);
  set_ = true;
  num_ = num;
}

bool SequenceNum::isSet() const {
  std::lock_guard<std::mutex> lock(lock_);
  return set_;
}

SequenceNum& SequenceNum::operator=(const SequenceNum& other) {
  if (!other.isSet()) {
    num_ = 0;
    set_ = false;
  } else {
    num_ = other.get();
    set_ = true;
  }
  return *this;
}

} // namespace c10d

#include <c10/util/Optional.h>
#include <ATen/ThreadLocalState.h>
#include <c10d/sequence_num.hpp>

#include <c10/util/Logging.h>

namespace c10d {
SequenceNum::SequenceNum() : num_(c10::nullopt) {}

SequenceNum::SequenceNum(const uint64_t num) : num_(num) {}

SequenceNum::SequenceNum(const SequenceNum& other) {
  num_.withLock([&](auto& num) {
    if (!other.isSet()) {
      num = c10::nullopt;
    } else {
      num = other.get();
    }
  });
}

uint64_t SequenceNum::get() const {
  uint64_t ret = 0;
  num_.withLock([&](const auto num) {
    TORCH_CHECK(num != c10::nullopt);
    ret = *num;
  });
  return ret;
}

void SequenceNum::increment() {
  num_.withLock([](auto& num) {
    TORCH_CHECK(num != c10::nullopt);
    num = ++(*num);
  });
}

// Implemented without above get() and increment() so we don't repeatedly lock
// and unblock.
uint64_t SequenceNum::getAndIncrement() {
  uint64_t curVal = 0;
  num_.withLock([&](auto& num) {
    TORCH_CHECK(num != c10::nullopt);
    curVal = *num;
    num = ++(*num);
  });
  return curVal;
}

void SequenceNum::set(const uint64_t num) {
  num_.withLock([&](auto& num_field) { num_field = num; });
}

bool SequenceNum::isSet() const {
  bool isSet = false;
  num_.withLock([&](const auto num) { isSet = num != c10::nullopt; });
  return isSet;
}

SequenceNum& SequenceNum::operator=(const SequenceNum& other) {
  num_.withLock([&](auto& num) {
    if (!other.isSet()) {
      num = c10::nullopt;
    } else {
      num = other.get();
    }
  });
  return *this;
}

} // namespace c10d

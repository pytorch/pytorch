#include <c10d/sequence_num.hpp>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

namespace c10d {
SequenceNum::SequenceNum() : num_(c10::nullopt) {}

SequenceNum::SequenceNum(const uint64_t num) : num_(num) {}

c10::optional<uint64_t> SequenceNum::num() const {
  return num_.withLock([](const auto num) { return num; });
}

SequenceNum::SequenceNum(const SequenceNum& other) : num_(other.num()) {}

uint64_t SequenceNum::get() const {
  return num_.withLock([](const auto num) {
    TORCH_CHECK(num != c10::nullopt);
    return *num;
  });
}

void SequenceNum::increment() {
  // Use the full type instead of `auto` to force a non-const reference.
  num_.withLock([](c10::optional<uint64_t>& num) {
    TORCH_CHECK(num != c10::nullopt);
    *num = *num + 1;
  });
}

// Implemented without above get() and increment() so we don't repeatedly lock
// and unlock.
uint64_t SequenceNum::getAndIncrement() {
  // Use the full type instead of `auto` to force a non-const reference.
  return num_.withLock([](c10::optional<uint64_t>& num) {
    TORCH_CHECK(num != c10::nullopt);
    uint64_t curVal = *num;
    *num = curVal + 1;
    return curVal;
  });
}

void SequenceNum::set(const uint64_t new_num) {
  // Use the full type instead of `auto` to force a non-const reference.
  num_.withLock([=](c10::optional<uint64_t>& num) { *num = new_num; });
}

bool SequenceNum::isSet() const {
  return num_.withLock([](const auto num) { return num != c10::nullopt; });
}

SequenceNum& SequenceNum::operator=(const SequenceNum& other) {
  // Use the full type instead of `auto` to force a non-const reference.
  num_.withLock([&](c10::optional<uint64_t>& num) { num = other.num(); });
  return *this;
}

} // namespace c10d

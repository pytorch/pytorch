#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
#include <mutex>
#include <optional>
#include <vector>

namespace c10d {
constexpr int kUnsetSeqNum = 0;

namespace {
constexpr int kByteOffset = 8;
} // namespace

// Converts from int to char vec to write in store
template <typename T>
inline std::vector<T> toVec(uint64_t num, int numBytes) {
  std::vector<T> values;
  // Read off bytes from right to left, pushing them into
  // char array.
  for (const auto i : c10::irange(numBytes)) {
    uint8_t x = (num >> (kByteOffset * i)) & 0xff;
    values.push_back(static_cast<T>(x));
  }
  return values;
}

// Converts from char vec (such as from store read) to int.
template <typename T>
inline uint64_t fromVec(const std::vector<T>& values) {
  uint64_t num = 0;
  // Set each byte at the correct location on num
  for (const auto i : c10::irange(values.size())) {
    uint8_t x = static_cast<uint8_t>(values[i]);
    num |= (static_cast<int64_t>(x) << (kByteOffset * i));
  }
  return num;
}

class TORCH_API SequenceNum {
 public:
  SequenceNum();
  explicit SequenceNum(const uint64_t num);
  // Retrieve num_. Will throw if not set.
  uint64_t get() const;
  // Increment num_. Will throw if not set.
  void increment();
  // Increment num_ and return the old value. Will throw if not set.
  uint64_t getAndIncrement();
  // Sets num_
  void set(const uint64_t num);
  // Returns true if this SequenceNum is properly initialized with a value, else
  // false.
  bool isSet() const;

  SequenceNum& operator=(const SequenceNum& other);

  SequenceNum(const SequenceNum& other);

 private:
  std::optional<uint64_t> num_;
  mutable std::mutex lock_;
};

} // namespace c10d

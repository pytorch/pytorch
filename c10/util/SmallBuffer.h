#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Helper class for allocating temporary fixed size arrays with SBO.
 *
 * This is intentionally much simpler than SmallVector, to improve performance
 * at the expense of many features:
 * - No zero-initialization for numeric types
 * - No resizing after construction
 * - No copy/move
 * - No non-trivial types
 */

namespace c10 {

template <typename T, size_t N>
class SmallBuffer {
  static_assert(std::is_trivial_v<T>, "SmallBuffer is intended for POD types");

  std::array<T, N> storage_;
  size_t size_{};
  T* data_{};

 public:
  SmallBuffer(size_t size) : size_(size) {
    if (size > N) {
      data_ = new T[size];
    } else {
      data_ = &storage_[0];
    }
  }

  SmallBuffer(const SmallBuffer&) = delete;
  SmallBuffer& operator=(const SmallBuffer&) = delete;

  // move constructor is needed in function return
  SmallBuffer(SmallBuffer&& rhs) noexcept : size_{rhs.size_} {
    rhs.size_ = 0;
    if (size_ > N) {
      data_ = rhs.data_;
      rhs.data_ = nullptr;
    } else {
      storage_ = std::move(rhs.storage_);
      data_ = &storage_[0];
    }
  }

  SmallBuffer& operator=(SmallBuffer&&) = delete;

  ~SmallBuffer() {
    if (size_ > N) {
      delete[] data_;
    }
  }
  T& operator[](size_t idx) {
    return data()[idx];
  }
  const T& operator[](size_t idx) const {
    return data()[idx];
  }
  T* data() {
    return data_;
  }
  const T* data() const {
    return data_;
  }
  size_t size() const {
    return size_;
  }
  T* begin() {
    return data_;
  }
  const T* begin() const {
    return data_;
  }
  T* end() {
    return data_ + size_;
  }
  const T* end() const {
    return data_ + size_;
  }
};

} // namespace c10

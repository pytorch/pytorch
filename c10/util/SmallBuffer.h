#pragma once
#include <type_traits>

/** Helper class for allocating temporary fixed size arrays with SBO.
 *
 * This is intentionally much simpler than SmallVector, to improve performace at
 * the expense of many features:
 * - No zero-initialization for numeric types
 * - No resizing after construction
 * - No copy/move
 * - No non-trivial types
 */

namespace c10 {

template <typename T, size_t N>
class SmallBuffer {
  static_assert(std::is_pod<T>::value, "SmallBuffer is intended for POD types");

  T storage_[N];
  size_t size_;
  T* data_;

 public:
  SmallBuffer(size_t size) : size_(size) {
    if (size > N) {
      data_ = new T[size];
    } else {
      data_ = &storage_[0];
    }
  }

  ~SmallBuffer() {
    if (size_ > N) {
      delete[] data_;
    }
  }

  T* data() {
    return data_;
  }
  size_t size() const {
    return size_;
  }
  T* begin() {
    return data_;
  }
  T* end() {
    return data_ + size_;
  }
};

} // namespace c10

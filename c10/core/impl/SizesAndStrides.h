#pragma once

#include <algorithm>
#include <cstdint>

#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>

namespace c10 {
namespace impl {

// Packed container for TensorImpl sizes and strides.
// This design improves on the previous approach of using a pair of
// c10::SmallVector<int64_t, 5> by specializing for the operations we
// actually use and enforcing that the number of sizes is the same as
// the number of strides. The memory layout is as follows:
//
// 1 size_t for the size
// 5 eightbytes of inline sizes and 5 eightbytes of inline strides, OR pointer to out-of-line array
class C10_API SizesAndStrides {
 public:
  // TODO: different iterator types for sizes & strides to prevent
  // mixing the two accidentally.
  using sizes_iterator = int64_t*;
  using sizes_const_iterator = const int64_t*;
  using strides_iterator = int64_t*;
  using strides_const_iterator = const int64_t*;

  SizesAndStrides() : size_(1) {
    size_at_unchecked(0) = 0;
    stride_at_unchecked(0) = 1;
  }

  ~SizesAndStrides() {
    if (C10_UNLIKELY(!isInline())) {
      freeOutOfLineStorage();
    }
  }

  SizesAndStrides(const SizesAndStrides& rhs) : size_(rhs.size_) {
    if (C10_LIKELY(rhs.isInline())) {
      copyDataInline(rhs);
    } else {
      allocateOutOfLineStorage(size_);
      copyDataOutline(rhs);
    }
  }

  SizesAndStrides& operator=(const SizesAndStrides& rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (C10_LIKELY(rhs.isInline())) {
      if (C10_UNLIKELY(!isInline())) {
        freeOutOfLineStorage();
      }
      copyDataInline(rhs);
    } else {
      if (isInline()) {
        allocateOutOfLineStorage(rhs.size_);
      } else {
        resizeOutOfLineStorage(rhs.size_);
      }
      copyDataOutline(rhs);
    }
    size_ = rhs.size_;
    return *this;
  }

  // Move from rhs. rhs.size() == 0 afterwards.
  SizesAndStrides(SizesAndStrides&& rhs) : size_(rhs.size_) {
    if (C10_LIKELY(isInline())) {
      memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
    } else {
      outOfLineStorage_ = rhs.outOfLineStorage_;
      rhs.outOfLineStorage_ = nullptr;
    }

    rhs.size_ = 0;
  }

  // Move from rhs. rhs.size() == 0 afterwards.
  SizesAndStrides& operator=(SizesAndStrides&& rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (C10_LIKELY(rhs.isInline())) {
      if (C10_UNLIKELY(!isInline())) {
        freeOutOfLineStorage();
      }
      copyDataInline(rhs);
    } else {
      // They're outline. We're going to steal their vector.
      if (!isInline()) {
        freeOutOfLineStorage();
      }
      outOfLineStorage_ = rhs.outOfLineStorage_;
      rhs.outOfLineStorage_ = nullptr;
    }
    size_ = rhs.size_;
    rhs.size_ = 0;

    return *this;
  }

  size_t size() const {
    return size_;
  }

  const int64_t* sizes_data() const {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return &outOfLineStorage_[0];
    }
  }

  int64_t* sizes_data() {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return &outOfLineStorage_[0];
    }
  }

  sizes_const_iterator sizes_begin() const {
    return sizes_data();
  }

  sizes_iterator sizes_begin()  {
    return sizes_data();
  }

  sizes_const_iterator sizes_end() const {
    return sizes_begin() + size();
  }

  sizes_iterator sizes_end() {
    return sizes_begin() + size();
  }

  IntArrayRef sizes_arrayref() const {
    return IntArrayRef{sizes_data(), size()};
  }

  void set_sizes(IntArrayRef newSizes) {
    resize(newSizes.size());
    std::copy(newSizes.begin(), newSizes.end(), sizes_begin());
  }

  const int64_t* strides_data() const {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  int64_t* strides_data() {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_const_iterator strides_begin() const {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_iterator strides_begin() {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_const_iterator strides_end() const {
    return strides_begin() + size();
  }

  strides_iterator strides_end() {
    return strides_begin() + size();
  }

  IntArrayRef strides_arrayref() const {
    return IntArrayRef{strides_data(), size()};
  }

  // Size accessors.
  int64_t size_at(size_t idx) const {
    assert(idx < size());
    return sizes_data()[idx];
  }

  int64_t& size_at(size_t idx) {
    assert(idx < size());
    return sizes_data()[idx];
  }

  int64_t size_at_unchecked(size_t idx) const {
    return sizes_data()[idx];
  }

  int64_t& size_at_unchecked(size_t idx) {
    return sizes_data()[idx];
  }

  // Size accessors.
  int64_t stride_at(size_t idx) const {
    assert(idx < size());
    return strides_data()[idx];
  }

  int64_t& stride_at(size_t idx) {
    assert(idx < size());
    return strides_data()[idx];
  }

  int64_t stride_at_unchecked(size_t idx) const {
    return strides_data()[idx];
  }

  int64_t& stride_at_unchecked(size_t idx) {
    return strides_data()[idx];
  }

  void resize(const size_t newSize) {
    const auto oldSize = size();
    if (newSize == oldSize) {
      return;
    }
    if (C10_LIKELY(newSize <= MAX_INLINE_SIZE)) {
      if (C10_LIKELY(isInline())) {
        if (oldSize < newSize) {
          const auto bytesToZero = (newSize - oldSize) * sizeof(inlineStorage_[0]);
          memset(&inlineStorage_[oldSize], 0, bytesToZero);
          memset(&inlineStorage_[MAX_INLINE_SIZE + oldSize], 0, bytesToZero);
        }
      } else {
        int64_t* tempStorage = outOfLineStorage_;
        memcpy(
            &inlineStorage_[0],
            &tempStorage[0],
            MAX_INLINE_SIZE * sizeof(inlineStorage_[0]));
        memcpy(
            &inlineStorage_[MAX_INLINE_SIZE],
            &tempStorage[oldSize],
            MAX_INLINE_SIZE * sizeof(inlineStorage_[0]));
        // CANNOT USE freeOutOfLineStorage() HERE! outOfLineStorage_
        // HAS BEEN OVERWRITTEN!
        free(tempStorage);
      }
    } else {
      if (isInline()) {
        // CANNOT USE allocateOutOfLineStorage(newSize) HERE! WOULD
        // OVERWRITE inlineStorage_!
        int64_t* tempStorage = static_cast<int64_t *>(malloc(storageBytes(newSize)));
        const auto bytesToCopy = oldSize * sizeof(inlineStorage_[0]);
        const auto bytesToZero = (newSize > oldSize) ? (newSize - oldSize) * sizeof(tempStorage[0]) : 0;
        memcpy(&tempStorage[0], &inlineStorage_[0], bytesToCopy);
        if (bytesToZero) {
          memset(&tempStorage[oldSize], 0, bytesToZero);
        }
        memcpy(&tempStorage[newSize], &inlineStorage_[MAX_INLINE_SIZE], bytesToCopy);
        if (bytesToZero) {
          memset(&tempStorage[newSize + oldSize], 0, bytesToZero);
        }
        outOfLineStorage_ = tempStorage;
      } else {
        const bool isGrowing = oldSize < newSize;
        if (isGrowing) {
          // Resize before shifting so that we have room.
          resizeOutOfLineStorage(newSize);
        }
        // Shift the old strides to their new starting point. Note
        // that this does not occur in the inline path above because
        // the stride starting point is not moving.
        memmove(
            outOfLineStorage_ + newSize,
            outOfLineStorage_ + oldSize,
            std::min(oldSize, newSize) * sizeof(outOfLineStorage_[0]));
        if (!isGrowing) {
          // Resize after shifting so that we don't lose data.
          resizeOutOfLineStorage(newSize);
        } else {
          // Zero the end of the sizes portion.
          const auto bytesToZero = (newSize - oldSize) * sizeof(outOfLineStorage_[0]);
          memset(&outOfLineStorage_[oldSize], 0, bytesToZero);
          memset(&outOfLineStorage_[newSize + oldSize], 0, bytesToZero);
        }
      }
    }
    size_ = newSize;
  }

 private:
  bool isInline() const {
    return size_ <= MAX_INLINE_SIZE;
  }

  void copyDataInline(const SizesAndStrides& rhs) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.isInline());
    memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
  }

  static size_t storageBytes(size_t size) {
    return size * 2 * sizeof(int64_t);
  }

  void allocateOutOfLineStorage(size_t size) {
    outOfLineStorage_ = static_cast<int64_t *>(malloc(storageBytes(size)));
  }

  void resizeOutOfLineStorage(size_t newSize) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isInline());
    outOfLineStorage_ = static_cast<int64_t *>(realloc(outOfLineStorage_, storageBytes(newSize)));
  }

  void freeOutOfLineStorage() {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isInline());
    free(outOfLineStorage_);
  }

  void copyDataOutline(const SizesAndStrides& rhs) {
    memcpy(outOfLineStorage_, rhs.outOfLineStorage_, storageBytes(rhs.size_));
  }

  static constexpr int MAX_INLINE_SIZE = 5;

  size_t size_;
  union {
    int64_t *outOfLineStorage_;
    int64_t inlineStorage_[MAX_INLINE_SIZE * 2];
  };

};

} // namespace impl
} // namespace c10

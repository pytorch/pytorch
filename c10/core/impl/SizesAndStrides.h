#pragma once

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
// 1 tagged intptr_t; if the tag bit is 1, the data is stored inline and
// the rest of the word indicates the size of the data. If the tag bit
// is 0, the tag word is a pointer to an array of int64_t holding the
// data; the first int64_t indicates the size of the rest.
// 5 int64_t reserved for inline sizes
// 5 int64_t reserved for inline strides
class C10_API SizesAndStrides {
 public:
  // TODO: different iterator types for sizes & strides to prevent
  // mixing the two accidentally.
  using sizes_iterator = int64_t*;
  using sizes_const_iterator = const int64_t*;
  using strides_iterator = int64_t*;
  using strides_const_iterator = const int64_t*;

  SizesAndStrides() {
    setInlineSize(1);
    size_at_unchecked(0) = 0;
    stride_at_unchecked(0) = 1;
  }

  ~SizesAndStrides() {
    if (C10_UNLIKELY(!isInline())) {
      delete outOfLineStorage();
    }
  }

  SizesAndStrides(const SizesAndStrides& rhs) {
    if (C10_LIKELY(rhs.isInline())) {
      copyFromInline(rhs);
    } else {
      taggedStorageOrSize_ = reinterpret_cast<int64_t>(new std::vector<int64_t>(*rhs.outOfLineStorage()));
    }
  }

  SizesAndStrides& operator=(const SizesAndStrides& rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (C10_LIKELY(rhs.isInline())) {
      if (C10_UNLIKELY(!isInline())) {
        delete outOfLineStorage();
      }
      copyFromInline(rhs);
    } else {
      if (isInline()) {
        taggedStorageOrSize_ = reinterpret_cast<int64_t>(new std::vector<int64_t>(*rhs.outOfLineStorage()));
      } else {
        // Both out of line. Copy their storage into ours.
        *outOfLineStorage() = *rhs.outOfLineStorage();
      }
    }
    return *this;
  }

  // Move from rhs. rhs.size() == 0 afterwards.
  SizesAndStrides(SizesAndStrides&& rhs) : taggedStorageOrSize_(rhs.taggedStorageOrSize_) {
    if (C10_LIKELY(isInline())) {
      memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
    }

    rhs.setInlineSize(0);
  }

  // Move from rhs. rhs.size() == 0 afterwards.
  SizesAndStrides& operator=(SizesAndStrides&& rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (C10_LIKELY(rhs.isInline())) {
      if (C10_UNLIKELY(!isInline())) {
        delete outOfLineStorage();
      }
      copyFromInline(rhs);
    } else {
      if (isInline()) {
        // Steal their vector.
        taggedStorageOrSize_ = rhs.taggedStorageOrSize_;
      } else {
        // Both out of line. Move their storage into ours.
        *outOfLineStorage() = std::move(*rhs.outOfLineStorage());
        delete rhs.outOfLineStorage();
      }
    }
    rhs.setInlineSize(0);

    return *this;
  }

  size_t size() const {
    if (C10_LIKELY(isInline())) {
      return taggedStorageOrSize_ >> 1;
    } else {
      return outOfLineStorage()->size() / 2;
    }
  }

  const int64_t* sizes_data() const {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return outOfLineStorage()->data();
    }
  }

  int64_t* sizes_data() {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return outOfLineStorage()->data();
    }
  }

  sizes_const_iterator sizes_begin() const {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return outOfLineStorage()->data();
    }
  }

  sizes_iterator sizes_begin()  {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return outOfLineStorage()->data();
    }
  }

  sizes_const_iterator sizes_end() const {
    return sizes_begin() + size();
  }

  sizes_iterator sizes_end() {
    return sizes_begin() + size();
  }

  const int64_t* strides_data() const {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[MAX_INLINE_SIZE];
    } else {
      return outOfLineStorage()->data() + size();
    }
  }

  int64_t* strides_data() {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[MAX_INLINE_SIZE];
    } else {
      return outOfLineStorage()->data() + size();
    }
  }

  strides_const_iterator strides_begin() const {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[MAX_INLINE_SIZE];
    } else {
      return outOfLineStorage()->data() + size();
    }
  }

  strides_iterator strides_begin() {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[MAX_INLINE_SIZE];
    } else {
      return outOfLineStorage()->data() + size();
    }
  }

  strides_const_iterator strides_end() const {
    return strides_begin() + size();
  }

  strides_iterator strides_end() {
    return strides_begin() + size();
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
      size_t oldInlineSize = 0;
      if (C10_LIKELY(isInline())) {
        oldInlineSize = oldSize;
        const auto bytesToZero = (newSize - oldInlineSize) * sizeof(inlineStorage_[0]);
        memset(&inlineStorage_[oldInlineSize], 0, bytesToZero);
        memset(&inlineStorage_[MAX_INLINE_SIZE + oldInlineSize], 0, bytesToZero);
      } else {
        memcpy(&inlineStorage_[0], outOfLineStorage()->data(), MAX_INLINE_SIZE * sizeof(inlineStorage_[0]));
        memcpy(
            &inlineStorage_[MAX_INLINE_SIZE],
            outOfLineStorage()->data() + outOfLineStorage()->size() / 2,
            MAX_INLINE_SIZE * sizeof(inlineStorage_[0]));
        delete outOfLineStorage();
      }
      setInlineSize(newSize);
    } else {
      if (isInline()) {
        taggedStorageOrSize_ = reinterpret_cast<int64_t>(new std::vector<int64_t>(newSize * 2));
        const auto bytesToCopy = oldSize * sizeof(inlineStorage_[0]);
        memcpy(outOfLineStorage()->data(), &inlineStorage_[0], bytesToCopy);
        memcpy(outOfLineStorage()->data() + newSize, &inlineStorage_[MAX_INLINE_SIZE], bytesToCopy);
      } else {
        const bool isGrowing = oldSize < newSize;
        if (isGrowing) {
          // Resize before shifting so that we have room.
          outOfLineStorage()->resize(newSize * 2);
        }
        // Shift the old strides to their new starting point. Note
        // that this does not occur in the inline path above because
        // the stride starting point is not moving.
        memmove(
            outOfLineStorage()->data() + newSize,
            outOfLineStorage()->data() + oldSize,
            std::min(oldSize, newSize) * sizeof(inlineStorage_[0]));
        if (!isGrowing) {
          // Resize after shifting so that we don't lose data.
          outOfLineStorage()->resize(newSize * 2);
        } else {
          // Zero the end of the sizes portion.
          memset(outOfLineStorage()->data() + oldSize, 0, (newSize - oldSize) * sizeof(inlineStorage_[0]));
        }
      }
    }
  }

 private:
  bool isInline() const {
    return (taggedStorageOrSize_ & 1) == 1;
  }

  std::vector<int64_t>* outOfLineStorage() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isInline());
    return reinterpret_cast<std::vector<int64_t>*>(taggedStorageOrSize_);
  }

  void resetToInlineStorage() {
    setInlineSize(1);
  }

  void setInlineSize(size_t sz) {
    taggedStorageOrSize_ = (sz << 1) | 1;
  }

  void copyFromInline(const SizesAndStrides& rhs) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.isInline());
    taggedStorageOrSize_ = rhs.taggedStorageOrSize_;
    memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
  }

  static constexpr int MAX_INLINE_SIZE = 5;
  int64_t taggedStorageOrSize_;
  int64_t inlineStorage_[MAX_INLINE_SIZE * 2];
};

} // namespace impl
} // namespace c10

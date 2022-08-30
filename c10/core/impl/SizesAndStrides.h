#pragma once

#include <algorithm>
#include <cstdint>

#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>

#define C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE 5

namespace c10 {
namespace impl {

// Packed container for TensorImpl sizes and strides.
// This design improves on the previous approach of using a pair of
// c10::SmallVector<int64_t, 5> by specializing for the operations we
// actually use and enforcing that the number of sizes is the same as
// the number of strides. The memory layout is as follows:
//
// 1 size_t for the size
// 5 eightbytes of inline sizes and 5 eightbytes of inline strides, OR pointer
// to out-of-line array
class C10_API SizesAndStrides {
 public:
  // TODO: different iterator types for sizes & strides to prevent
  // mixing the two accidentally.
  using sizes_iterator = SymInt*;
  using sizes_const_iterator = const SymInt*;
  using strides_iterator = SymInt*;
  using strides_const_iterator = const SymInt*;

  SizesAndStrides() : size_(1) {
    size_at_unchecked(0) = 0;
    stride_at_unchecked(0) = 1;
  }

  ~SizesAndStrides() {
    if (C10_UNLIKELY(!isInline())) {
      delete[] outOfLineStorage_;
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
        delete[] outOfLineStorage_;
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
  SizesAndStrides(SizesAndStrides&& rhs) noexcept : size_(rhs.size_) {
    if (C10_LIKELY(isInline())) {
      for (size_t i = 0; i < sizeof(inlineStorage_) / sizeof(SymInt); i++) {
        inlineStorage_[i] = std::move(rhs.inlineStorage_[i]);
      }
    } else {
      outOfLineStorage_ = rhs.outOfLineStorage_;
      rhs.outOfLineStorage_ = nullptr;
    }

    rhs.size_ = 0;
  }

  // Move from rhs. rhs.size() == 0 afterwards.
  SizesAndStrides& operator=(SizesAndStrides&& rhs) noexcept {
    if (this == &rhs) {
      return *this;
    }
    if (C10_LIKELY(rhs.isInline())) {
      if (C10_UNLIKELY(!isInline())) {
        delete[] outOfLineStorage_;
      }
      copyDataInline(rhs);
    } else {
      // They're outline. We're going to steal their vector.
      if (!isInline()) {
        delete[] outOfLineStorage_;
      }
      outOfLineStorage_ = rhs.outOfLineStorage_;
      rhs.outOfLineStorage_ = nullptr;
    }
    size_ = rhs.size_;
    rhs.size_ = 0;

    return *this;
  }

  size_t size() const noexcept {
    return size_;
  }

  const SymInt* sizes_data() const noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return &outOfLineStorage_[0];
    }
  }

  bool has_sym_slow() const noexcept {
    if (std::any_of(sizes_begin(), sizes_end(), [](const auto i) {
          return i.is_symbolic();
        })) {
      return true;
    }

    if (std::any_of(strides_begin(), strides_end(), [](const auto i) {
          return i.is_symbolic();
        })) {
      return true;
    }

    return false;
  }

  SymInt* sizes_data() noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return &outOfLineStorage_[0];
    }
  }

  sizes_const_iterator sizes_begin() const noexcept {
    return sizes_data();
  }

  sizes_iterator sizes_begin() noexcept {
    return sizes_data();
  }

  sizes_const_iterator sizes_end() const noexcept {
    return sizes_begin() + size();
  }

  sizes_iterator sizes_end() noexcept {
    return sizes_begin() + size();
  }

  SymIntArrayRef sizes_arrayref() const noexcept {
    return SymIntArrayRef{sizes_data(), size()};
  }

  void set_sizes(SymIntArrayRef newSizes) {
    resize(newSizes.size());
    std::copy(newSizes.begin(), newSizes.end(), sizes_begin());
  }

  void set_strides(SymIntArrayRef strides) {
    TORCH_INTERNAL_ASSERT(strides.size() == size());
    std::copy(strides.begin(), strides.end(), strides_begin());
  }

  void set_sizes(IntArrayRef newSizes) {
    set_sizes(SymIntArrayRef::fromIntArrayRef(newSizes));
  }

  const SymInt* strides_data() const noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  SymInt* strides_data() noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_const_iterator strides_begin() const noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_iterator strides_begin() noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_const_iterator strides_end() const noexcept {
    return strides_begin() + size();
  }

  strides_iterator strides_end() noexcept {
    return strides_begin() + size();
  }

  SymIntArrayRef strides_arrayref() const noexcept {
    return SymIntArrayRef{strides_data(), size()};
  }

  // Size accessors.
  SymInt size_at(size_t idx) const noexcept {
    assert(idx < size());
    return sizes_data()[idx];
  }

  SymInt& size_at(size_t idx) noexcept {
    assert(idx < size());
    return sizes_data()[idx];
  }

  SymInt size_at_unchecked(size_t idx) const noexcept {
    return sizes_data()[idx];
  }

  SymInt& size_at_unchecked(size_t idx) noexcept {
    return sizes_data()[idx];
  }

  // Size accessors.
  SymInt stride_at(size_t idx) const noexcept {
    assert(idx < size());
    return strides_data()[idx];
  }

  SymInt& stride_at(size_t idx) noexcept {
    assert(idx < size());
    return strides_data()[idx];
  }

  SymInt stride_at_unchecked(size_t idx) const noexcept {
    return strides_data()[idx];
  }

  SymInt& stride_at_unchecked(size_t idx) noexcept {
    return strides_data()[idx];
  }

  void resize(size_t newSize) {
    const auto oldSize = size();
    if (newSize == oldSize) {
      return;
    }
    if (C10_LIKELY(
            newSize <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE && isInline())) {
      if (oldSize < newSize) {
        for (size_t i = oldSize; i < newSize; i++) {
          inlineStorage_[i] = 0;
          inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE + i] = 0;
        }
      }
      size_ = newSize;
    } else {
      resizeSlowPath(newSize, oldSize);
    }
  }

  void resizeSlowPath(size_t newSize, size_t oldSize);

 private:
  bool isInline() const noexcept {
    return size_ <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE;
  }

  void copyDataInline(const SizesAndStrides& rhs) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.isInline());
    for (size_t i = 0; i < sizeof(inlineStorage_) / sizeof(SymInt); i++) {
      inlineStorage_[i] = rhs.inlineStorage_[i];
    }
  }

  static size_t storageElems(size_t size) noexcept {
    return size * 2;
  }

  void allocateOutOfLineStorage(size_t size) {
    outOfLineStorage_ = new SymInt[storageElems(size)];
    TORCH_CHECK(
        outOfLineStorage_,
        "Could not allocate memory for Tensor SizesAndStrides!");
  }

  void resizeOutOfLineStorage(size_t newSize) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isInline());
    auto* newStorage = new SymInt[storageElems(newSize)];
    TORCH_CHECK(
        newStorage, "Could not allocate memory for Tensor SizesAndStrides!");
    for (size_t i = 0; i < storageElems(newSize) && i < storageElems(size_);
         i++) {
      newStorage[i] = std::move(outOfLineStorage_[i]);
    }
    delete[] outOfLineStorage_;
    outOfLineStorage_ = newStorage;
  }

  void copyDataOutline(const SizesAndStrides& rhs) noexcept {
    for (size_t i = 0; i < storageElems(rhs.size_); i++) {
      outOfLineStorage_[i] = rhs.outOfLineStorage_[i];
    }
  }

  size_t size_;
  union {
    SymInt* outOfLineStorage_;
    SymInt inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * 2]{};
  };
};

} // namespace impl
} // namespace c10

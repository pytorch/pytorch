#pragma once

#include <algorithm>

#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>

namespace c10 {
namespace impl {

// Packed container for TensorImpl sizes and strides.
class C10_API SizesAndStrides {
 public:
  // TODO: different iterator types for sizes & strides to prevent
  // mixing the two accidentally.
  using sizes_iterator = int64_t*;
  using sizes_const_iterator = const int64_t*;
  using strides_iterator = int64_t*;
  using strides_const_iterator = const int64_t*;

  SizesAndStrides() : sizes_{0}, strides_{1} {}

  size_t size() const {
    return sizes_.size();
  }

  const int64_t* sizes_data() const {
    return sizes_.data();
  }

  int64_t* sizes_data() {
    return sizes_.data();
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
    return strides_.data();
  }

  int64_t* strides_data() {
    return strides_.data();
  }

  strides_const_iterator strides_begin() const {
    return strides_data();
  }

  strides_iterator strides_begin() {
    return strides_data();
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
    return sizes_.at(idx);
  }

  int64_t& size_at(size_t idx) {
    return sizes_.at(idx);
  }

  int64_t size_at_unchecked(size_t idx) const {
    return sizes_[idx];
  }

  int64_t& size_at_unchecked(size_t idx) {
    return sizes_[idx];
  }

  // Size accessors.
  int64_t stride_at(size_t idx) const {
    return strides_.at(idx);
  }

  int64_t& stride_at(size_t idx) {
    return strides_.at(idx);
  }

  int64_t stride_at_unchecked(size_t idx) const {
    return strides_[idx];
  }

  int64_t& stride_at_unchecked(size_t idx) {
    return strides_[idx];
  }

  void resize(size_t sz) {
    sizes_.resize(sz);
    strides_.resize(sz);
  }

 private:
  SmallVector<int64_t,5> sizes_;
  SmallVector<int64_t,5> strides_;
};

} // namespace impl
} // namespace c10

#pragma once

#include <string>
#include <vector>

#include "torch/csrc/jit/tensorexpr/eval.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename T>
struct DefaultPaddedValue;

template <>
struct DefaultPaddedValue<int> {
  static const int kValue = static_cast<int>(0xDEADBEEF);
};

template <>
struct DefaultPaddedValue<float> {
  static constexpr float kValue = 0.1357;
};

// A concrete base to be used in PaddedBase.
class PaddedBufferBase {
 public:
  const std::string& name() const {
    return name_;
  }

  int size() const {
    return total_size_;
  }

  int raw_size() const {
    return total_size_ + 2 * kPaddingSize;
  }

  virtual ~PaddedBufferBase() {}

 protected:
  explicit PaddedBufferBase(
      const std::vector<int>& dims,
      const std::string& name);
  int Index(const std::vector<int>& indices) const;

  std::vector<int> dims_;
  std::string name_;
  std::vector<int> strides_;
  int total_size_; // total number of useful element, does not include the
                   // paddings
  static constexpr int kPaddingSize = 64;
};

// A padded buffer with wartermarks for testing.
// The buffer carries padded watermarks on both sides to catch potential
// out-of-bounds writes. For read-only data that are not supposed to change, it
// can also make a backup and be compared later.
template <typename T>
class PaddedBuffer : public PaddedBufferBase {
 public:
  PaddedBuffer(int d0, const std::string& name = "")
      : PaddedBuffer(std::vector<int>({d0}), name) {}
  PaddedBuffer(int d0, int d1, const std::string& name = "")
      : PaddedBuffer(std::vector<int>({d0, d1}), name) {}
  PaddedBuffer(int d0, int d1, int d2, const std::string& name = "")
      : PaddedBuffer(std::vector<int>({d0, d1, d2}), name) {}
  PaddedBuffer(int d0, int d1, int d2, int d3, const std::string& name = "")
      : PaddedBuffer(std::vector<int>({d0, d1, d2, d3}), name) {}
  PaddedBuffer(const std::vector<int>& dims, const std::string& name = "")
      : PaddedBufferBase(dims, name) {
    data_.resize(total_size_ + 2 * kPaddingSize, kPaddingValue);
  }
  PaddedBuffer(const PaddedBuffer& other, const std::string& name)
      : PaddedBuffer(other) {
    this->name_ = name;
  }

  T* data() {
    return data_.data() + kPaddingSize;
  }
  const T* data() const {
    return const_cast<PaddedBuffer*>(this)->data();
  }
  T* raw_data() {
    return data_.data();
  }
  const T* raw_data() const {
    return const_cast<PaddedBuffer*>(this)->raw_data();
  }
  T& operator()(int i0) {
    // There is a bit performance impact with forming a vector here. But this
    // data structure is for testing only, and not performance critical.
    return this->operator()(std::vector<int>({i0}));
  }
  const T& operator()(int i0) const {
    return const_cast<PaddedBuffer*>(this)->operator()(i0);
  }
  T& operator()(int i0, int i1) {
    return this->operator()(std::vector<int>({i0, i1}));
  }
  const T& operator()(int i0, int i1) const {
    return const_cast<PaddedBuffer*>(this)->operator()(i0, i1);
  }
  T& operator()(int i0, int i1, int i2) {
    return this->operator()(std::vector<int>({i0, i1, i2}));
  }
  const T& operator()(int i0, int i1, int i2) const {
    return const_cast<PaddedBuffer*>(this)->operator()(i0, i1, i2);
  }
  T& operator()(int i0, int i1, int i2, int i3) {
    return this->operator()(std::vector<int>({i0, i1, i2, i3}));
  }
  const T& operator()(int i0, int i1, int i2, int i3) const {
    return const_cast<PaddedBuffer*>(this)->operator()(i0, i1, i2, i3);
  }
  T& operator()(const std::vector<int>& indices) {
    return data_[kPaddingSize + Index(indices)];
  }
  const T& operator()(const std::vector<int>& indices) const {
    return const_cast<PaddedBuffer*>(this)->operator()(indices);
  }

  friend void ExpectAllNear(
      const PaddedBuffer<float>& v1,
      const PaddedBuffer<float>& v2,
      float abs_error);
  template <typename U>
  friend void ExpectAllEqual(
      const PaddedBuffer<U>& v1,
      const PaddedBuffer<U>& v2);
  // Verify the watermarks in the paddings are intact.
  void ValidateWatermark() const;
  void Backup() {
    backup_data_ = data_;
  }
  void CheckBackup() const;

 private:
  std::vector<T> data_;
  std::vector<T> backup_data_;
  T kPaddingValue = DefaultPaddedValue<T>::kValue;
};

template <typename T>
inline CodeGen::CallArg::CallArg(const PaddedBuffer<T>& buffer)
    : ptr_(const_cast<T*>(buffer.data())) {}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

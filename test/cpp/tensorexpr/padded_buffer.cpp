#include "test/cpp/tensorexpr/padded_buffer.h"

#include <sstream>

#include <gtest/gtest.h>

#include <c10/util/Logging.h>

namespace torch {
namespace jit {
namespace tensorexpr {

int PaddedBufferBase::Index(const std::vector<int>& indices) const {
  DCHECK_EQ(dims_.size(), indices.size());
  int total_index = 0;
  for (int i = 0; i < dims_.size(); i++) {
    total_index += indices[i] * strides_[i];
  }
  return total_index;
}

PaddedBufferBase::PaddedBufferBase(
    const std::vector<int>& dims,
    const std::string& name)
    : dims_(dims), name_(name), strides_(dims.size()) {
  for (int i = dims.size() - 1; i >= 0; --i) {
    if (i == dims.size() - 1) {
      strides_[i] = 1;
    } else {
      strides_[i] = strides_[i + 1] * dims[i + 1];
    }
  }
  total_size_ = strides_[0] * dims[0];
}

template <typename T>
std::string CompareErrorMsg(
    const PaddedBuffer<T>& v1,
    const PaddedBuffer<T>& v2,
    int index) {
  std::ostringstream oss;
  oss << "index: " << index << ", names: " << v1.name() << ", " << v2.name();
  return oss.str();
}

template <typename T>
void PaddedBuffer<T>::ValidateWatermark() const {
  for (int i = 0; i < kPaddingSize; i++) {
    EXPECT_EQ(data_[i], kPaddingValue)
        << "left-side watermark broken: "
        << "index: " << i << ", name: " << name();
    EXPECT_EQ(data_[i + total_size_ + kPaddingSize], kPaddingValue)
        << "right-side watermark broken: "
        << "index: " << i << ", name: " << name();
  }
}

template <typename T>
void PaddedBuffer<T>::CheckBackup() const {
  ValidateWatermark();
  DCHECK(backup_data_.size() == data_.size())
      << "Please make sure you have call Backup() before calling CheckBackup()";
  for (int i = 0; i < total_size_; i++) {
    EXPECT_EQ(data_[i + kPaddingSize], backup_data_[i + kPaddingSize])
        << "mismatch against backup, "
        << "index: " << i << ", name: " << name();
  }
}

template <typename T>
void ExpectAllEqual(const PaddedBuffer<T>& f1, const PaddedBuffer<T>& f2) {
  const std::vector<T>& v1 = f1.data_;
  const std::vector<T>& v2 = f2.data_;
  const int kPaddingSize = f1.kPaddingSize;
  const int total_size = f1.total_size_;
  ASSERT_EQ(v1.size(), v2.size());
  f1.ValidateWatermark();
  f2.ValidateWatermark();
  for (int i = 0; i < total_size; i++) {
    EXPECT_EQ(v1[kPaddingSize + i], v2[kPaddingSize + i])
        << CompareErrorMsg(f1, f2, i);
  }
}

void ExpectAllNear(
    const PaddedBuffer<float>& f1,
    const PaddedBuffer<float>& f2,
    float abs_error) {
  const std::vector<float>& v1 = f1.data_;
  const std::vector<float>& v2 = f2.data_;
  const int kPaddingSize = f1.kPaddingSize;
  const int total_size = f1.total_size_;
  ASSERT_EQ(v1.size(), v2.size());
  f1.ValidateWatermark();
  f2.ValidateWatermark();
  for (int i = 0; i < total_size; i++) {
    EXPECT_NEAR(v1[kPaddingSize + i], v2[kPaddingSize + i], abs_error)
        << CompareErrorMsg(f1, f2, i);
  }
}

template class PaddedBuffer<int>;
template class PaddedBuffer<float>;
template void ExpectAllEqual(
    const PaddedBuffer<int>& f1,
    const PaddedBuffer<int>& f2);

} // namespace tensorexpr
} // namespace jit
} // namespace torch

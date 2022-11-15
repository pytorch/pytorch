#ifndef CAFFE2_CORE_QTENSOR_H_
#define CAFFE2_CORE_QTENSOR_H_

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <c10/util/typeid.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

namespace caffe2 {

template <class Context>
class C10_EXPORT QTensor {
 public:
  QTensor() {}
  virtual ~QTensor() {}
  /**
   * @brief Creates a quantized tensor of the given dimension.
   *
   * Note that the actual data allocation is not going to be carried out until
   * the first time mutable_data() is called.
   *
   * The underlying storage of the quantized tensor interleaves elements
   * by bit depth.
   *
   * Labeled memory for tensor of size 6, precision 3
   *   [ E1[0] E2[0] E3[0] E4[0] E5[0] E6[0] ] // Least significant Bits
   *   [ E1[1] E2[1] E3[1] E4[1] E5[1] E6[1] ]
   *   [ E1[2] E2[2] E3[2] E4[2] E5[2] E6[2] ]
   *
   * In the case of sign bits (see enable_sign argument), an extra bit
   * per element is added:
   *
   * Labeled memory for tensor of size 6, precision 3, sign bit enabled
   *   [ E1[0] E2[0] E3[0] E4[0] E5[0] E6[0] ]
   *   [ E1[1] E2[1] E3[1] E4[1] E5[1] E6[1] ]
   *   [ E1[2] E2[2] E3[2] E4[2] E5[2] E6[2] ]
   *   [ E1[s] E2[s] E3[s] E4[s] E5[s] E6[s] ]
   *   Where 's' is 1 if E is negative
   *
   * The reason for this layout is the ability to efficiently multiply
   * many low precision integers as a sum of popcnt(A & B) * 1 << bit.
   * Explained here: https://arxiv.org/abs/1606.06160
   */
  // TODO: changing at::ArrayRef<int> to at::ArrayRef<int64_t>?
  explicit QTensor(
      at::ArrayRef<int> dims,
      const unsigned char precision,
      const bool signbit = false)
      : precision_(precision), signed_(signbit) {
    Resize(dims);
  }

  void Resize(at::ArrayRef<int> dim_source) {
    if (dims_ != dim_source) {
      const auto source_size = c10::multiply_integers(dim_source);
      if (static_cast<size_t>(source_size * (precision_ + signed_)) > capacity_) {
        data_ptr_.clear();
        capacity_ = 0;
      }
      dims_ = dim_source.vec();
      size_ = source_size;
    }
  }

  void
  SetBitAtIndex(const unsigned char bit, const size_t index, const bool value) {
    // Get the mutable data at bit depth `bit`.
    unsigned char* d = mutable_data();

    CAFFE_ENFORCE(
        bit < precision_ + signed_,
        "Attempted to a set a bit that is not allocated.");
    CAFFE_ENFORCE(bit * aligned_size() < capacity_);

    auto idx = (aligned_size() * bit) / CHAR_BIT;
    d = &d[idx];

    idx = index / CHAR_BIT;
    auto shift = CHAR_BIT - (index % CHAR_BIT) - 1;

    if (value) {
      d[idx] |= 1 << shift;
    } else {
      d[idx] &= ~(1 << shift);
    }
  }

  bool GetBitAtIndex(const unsigned char bit, const size_t index) const {
    // Get the data at bit depth `bit`
    const unsigned char* d = data();
    auto idx = (aligned_size() * bit) / CHAR_BIT;
    d = &d[idx];

    idx = index / CHAR_BIT;
    auto shift = CHAR_BIT - (index % CHAR_BIT) - 1;

    return d[idx] & (1 << shift);
  }

  void SetPrecision(const unsigned char precision) {
    precision_ = precision;
    data_ptr_.clear();
  }

  void SetSigned(const bool make_signed = true) {
    signed_ = make_signed;
    data_ptr_.clear();
  }

  void SetScale(const double scale) {
    scale_ = scale;
  }

  void SetBias(const double bias) {
    bias_ = bias;
  }

  unsigned char* mutable_data() {
    if (!data_ptr_) {
      data_ptr_ = Context::New(nbytes());
      capacity_ = nbytes() * CHAR_BIT;
    }
    CAFFE_ENFORCE(capacity_ == nbytes() * CHAR_BIT);
    return static_cast<unsigned char*>(data_ptr_.get());
  }

  inline const unsigned char* data() const {
    return static_cast<unsigned char*>(data_ptr_.get());
  }

  inline size_t size() const {
    return size_;
  }

  inline unsigned char alignment() const {
    return alignment_;
  }

  inline unsigned char precision() const {
    return precision_;
  }

  inline at::ArrayRef<int> sizes() const {
    return dims_;
  }

  // TODO: deprecate?
  inline at::ArrayRef<int> dims() const {
    return dims_;
  }

  inline bool is_signed() const {
    return signed_;
  }

  /**
   * Returns the number of dimensions of the data.
   */
  inline int ndim() const {
    return dims_.size();
  }

  inline size_t aligned_size() const {
    return alignment_ * ((size_ + alignment_ - 1) / alignment_);
  }

  inline size_t nbytes() const {
    return (aligned_size() * (precision_ + signed_)) / CHAR_BIT;
  }

  inline double scale() const {
    return scale_;
  }

  inline double bias() const {
    return bias_;
  }

  /**
   * Returns the i-th dimension of the qtensor in int.
   */
  inline int dim32(const int i) const {
    TORCH_DCHECK_LT(i, static_cast<int>(dims_.size())) << "Exceeding ndim limit " << dims_.size();
    TORCH_DCHECK_GE(i, 0) << "Cannot have negative index";
    CAFFE_ENFORCE_LT(dims_[i], std::numeric_limits<int>::max());
    return static_cast<int>(dims_[i]);
  }

  /**
   * Returns the 'canonical' version of a (usually)  user-specified axis,
   * allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < ndim(), return index.
   *        If -ndim <= index <= -1, return (ndim() - (-index)),
   *        e.g., the last axis index (ndim() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int canonical_axis_index(int axis_index) const {
    CAFFE_ENFORCE_GE(axis_index, -ndim());
    CAFFE_ENFORCE_LT(axis_index, ndim());
    if (axis_index < 0) {
      return axis_index + ndim();
    }
    return axis_index;
  }

  /**
   * Return product of all dimensions starting from K.
   */
  inline int64_t size_from_dim(int k) const {
    int64_t r = 1;
    for (const auto i : c10::irange(k, dims_.size())) {
      r *= dims_[i];
    }
    return r;
  }

  /**
   * Product of all dims up to.
   */
  inline int64_t size_to_dim(int k) const {
    CAFFE_ENFORCE(k < dims_.size());
    int64_t r = 1;
    for (const auto i : c10::irange(k)) {
      r *= dims_[i];
    }
    return r;
  }

 protected:
  std::vector<int> dims_;
  size_t size_ = 0;

  // Precision in bits.
  unsigned char precision_ = CHAR_BIT;
  // Bit alignment.
  unsigned char alignment_ = CHAR_BIT;

  // Allocated data.
  at::DataPtr data_ptr_;

  // value = scale_ * (x + bias_)
  double scale_;
  double bias_;
  bool signed_ = false;

  // Capacity in bits.
  size_t capacity_ = 0;
};

} // namespace caffe2
#endif // CAFFE2_CORE_QTENSOR_H_

#pragma once
#include <ATen/core/ivalue_to.h>
#include <c10/util/ArrayRef.h>
#include <caffe2/core/tensor.h>


namespace c10 {
struct IValue;
}

namespace caffe2 {
namespace detail {

TORCH_API caffe2::Tensor contiguous(caffe2::Tensor);

/* This class represents input arguments to a caffe2 operator when
 * called from c10. It's purpose is to act as a compilation barrier
 * between caffe2 and ATen, so that caffe2 operators don't depend on
 * ATen operators implicitly through c10::IValue.
 */

class TORCH_API IValueInterface {
public:
  IValueInterface(): values_(nullptr), size_(0) {}
  explicit IValueInterface(c10::ArrayRef<c10::IValue> values);
  ~IValueInterface();

  bool isTensorList(size_t idx) const;
  bool isTensor(size_t idx) const;

  size_t size() const {
    return size_;
  }
  bool empty() const {
    return size() == 0;
  }

  const c10::IValue& at(size_t idx) const;

  // Expects isTensorList(idx)
  std::vector<caffe2::Tensor> toTensorVector(size_t idx) const;
  // Expects isTensor(idx)
  caffe2::Tensor toTensor(size_t idx) const;

  // Returns the number of tensor inputs, handling the case where the
  // inputs are passed as a TensorList.
  int compute_input_size() const;

  // These template functions are not defined in the header,
  // but are explicitly instantiated in IValueInterface.cc
  template <typename T>
  typename c10::detail::ivalue_to_const_ref_overload_return<T>::type to(size_t idx) const;

  template <typename T>
  std::vector<T> toVec(size_t idx) const;

private:
  c10::IValue *values_;
  size_t size_;
};

}}  // namespace caffe2::detail

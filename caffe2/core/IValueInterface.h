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

class TORCH_API IValueInterface {
public:
  IValueInterface(): values_(nullptr), size_(0) {}
  IValueInterface(ArrayRef<c10::IValue> values);
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
  std::vector<caffe2::Tensor> toTensorVector(size_t idx) const;
  caffe2::Tensor toTensor(size_t idx) const;

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

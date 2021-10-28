#define TORCH_ASSERT_NO_OPERATORS
#include <caffe2/core/IValueInterface.h>
#undef TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/ivalue.h>
#include <c10/util/irange.h>

#include <memory>

namespace caffe2 {
namespace detail {

caffe2::Tensor contiguous(caffe2::Tensor t) {
  return caffe2::Tensor(at::Tensor(std::move(t)).contiguous());
}


IValueInterface::IValueInterface(c10::ArrayRef<c10::IValue> values):
  size_(values.size()) {
  auto values_tmp = std::make_unique<c10::IValue[]>(size_);
  std::copy_n(values.data(), size_, values_tmp.get());
  values_ = values_tmp.release();
}

IValueInterface::~IValueInterface() {
  delete[] values_;
}

bool IValueInterface::isTensorList(size_t idx) const {
  return at(idx).isTensorList();
}

bool IValueInterface::isTensor(size_t idx) const {
  return at(idx).isTensor();
}

const c10::IValue& IValueInterface::at(size_t idx) const {
  TORCH_INTERNAL_ASSERT(idx < size_, "caffe2: Input index ", idx,
                        " is out of range for call with ", size_, " arguments");
  return values_[idx];
}

std::vector<caffe2::Tensor> IValueInterface::toTensorVector(size_t idx) const {
  auto list = at(idx).toTensorList();
  std::vector<caffe2::Tensor> ret(list.size());
  for (auto i : c10::irange(list.size())) {
    ret[i] = caffe2::Tensor(list.get(i));
  }
  return ret;
}

caffe2::Tensor IValueInterface::toTensor(size_t idx) const {
  return caffe2::Tensor(at(idx).toTensor());
}

int IValueInterface::compute_input_size() const {
  if (empty()) {
    return 0;
  }
  if (values_[0].isTensorList()) {
    // if the first input is a tensor list, we get input tensors by indexing
    // into that list. currently, this means that only tensors from that list
    // are accessible as inputs. any hypothetical input tensors that come after
    // the list are not accessible.
    return values_[0].toTensorList().size();
  }
  // it's not a tensor list. Count the number of tensor inputs and return them.
  int num_tensor_inputs = 0;
  bool found_nontensor = false;
  for (auto i: c10::irange(size())) {
    if (values_[i].isTensor()) {
      TORCH_INTERNAL_ASSERT(
          !found_nontensor,
          "All tensor arguments must come before non-tensor arguments");
      ++num_tensor_inputs;
    } else {
      found_nontensor = true;
    }
  }
  return num_tensor_inputs;
}

template <typename T>
typename c10::detail::ivalue_to_const_ref_overload_return<T>::type
IValueInterface::to(size_t idx) const {
  return at(idx).template to<T>();
}

#define INSTANTIATE_TO(Type) \
  template typename c10::detail::ivalue_to_const_ref_overload_return<Type>::type \
  IValueInterface::to<Type>(size_t idx) const

INSTANTIATE_TO(at::Tensor);
INSTANTIATE_TO(at::Storage);
INSTANTIATE_TO(c10::Stream);
INSTANTIATE_TO(float);
INSTANTIATE_TO(double);
INSTANTIATE_TO(c10::complex<double>);
INSTANTIATE_TO(unsigned char);
INSTANTIATE_TO(signed char);
INSTANTIATE_TO(unsigned short);
INSTANTIATE_TO(short);
INSTANTIATE_TO(int);
INSTANTIATE_TO(uint32_t);
INSTANTIATE_TO(uint64_t);
INSTANTIATE_TO(int64_t);
INSTANTIATE_TO(bool);
INSTANTIATE_TO(c10::intrusive_ptr<caffe2::Blob>);
INSTANTIATE_TO(c10::intrusive_ptr<ivalue::ConstantString>);
INSTANTIATE_TO(c10::intrusive_ptr<ivalue::Object>);
INSTANTIATE_TO(at::Scalar);
INSTANTIATE_TO(c10::List<int64_t>);
INSTANTIATE_TO(c10::List<double>);
INSTANTIATE_TO(c10::List<c10::complex<double>>);
INSTANTIATE_TO(c10::List<bool>);
INSTANTIATE_TO(c10::List<at::Tensor>);
INSTANTIATE_TO(c10::impl::GenericList);
INSTANTIATE_TO(c10::impl::GenericDict);
INSTANTIATE_TO(c10::intrusive_ptr<ivalue::Tuple>);
INSTANTIATE_TO(std::string);
INSTANTIATE_TO(c10::string_view);
INSTANTIATE_TO(c10::intrusive_ptr<ivalue::Future>);
INSTANTIATE_TO(c10::intrusive_ptr<c10::RRefInterface>);
INSTANTIATE_TO(c10::intrusive_ptr<at::Quantizer>);
INSTANTIATE_TO(IValue);
INSTANTIATE_TO(c10::Device);
INSTANTIATE_TO(at::ScalarType);
INSTANTIATE_TO(at::Layout);
INSTANTIATE_TO(at::MemoryFormat);
INSTANTIATE_TO(at::QScheme);
INSTANTIATE_TO(at::Dimname);
INSTANTIATE_TO(at::Generator);

template <typename T>
struct list_value_type {
  using type = T;
};

template <>
struct list_value_type<int> {
    using type = int64_t;
};

template <>
struct list_value_type<int16_t> {
  using type = int64_t;
};

template <>
struct list_value_type<float> {
  using type = double;
};

template <typename T, typename U>
static std::vector<T> to_vector(const c10::List<U> &list) {
  std::vector<T> ret;
  for (auto i: c10::irange(list.size())) {
    ret.push_back(list.get(i));
  }
  return ret;
}

template <typename T>
std::vector<T> IValueInterface::toVec(size_t idx) const {
  using list_value_t = typename list_value_type<T>::type;
  return to_vector<T>(to<c10::List<list_value_t>>(idx));
}

#define INSTANTIATE_TO_VEC(Type)                                        \
  template std::vector<Type> IValueInterface::toVec<Type>(size_t idx) const;

INSTANTIATE_TO_VEC(int64_t);
INSTANTIATE_TO_VEC(int32_t);
INSTANTIATE_TO_VEC(int16_t);
INSTANTIATE_TO_VEC(float);
INSTANTIATE_TO_VEC(double);
INSTANTIATE_TO_VEC(bool);
INSTANTIATE_TO_VEC(std::string);

}}  // namespace caffe2::detail

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "base/tensors/generic/THTensor.cpp"
#else

template<>
THTensor<real>::THTensor():
  tensor(THTensor_(new)())
  {};

template<>
THTensor<real>::THTensor(THRealTensor *wrapped):
  tensor(wrapped)
  {};

template<>
THTensor<real>::~THTensor() {
  if (tensor)
    THTensor_(free)(tensor);
}

template<>
auto THTensor<real>::clone() const -> THTensor* {
  return new THTensor(THTensor_(newClone)(tensor));
}

template<>
int THTensor<real>::nDim() const {
  return tensor->nDimension;
}

template<>
auto THTensor<real>::sizes() const -> long_range {
  return std::vector<long>(tensor->size, tensor->size + tensor->nDimension);
}

template<>
const long* THTensor<real>::rawSizes() const {
  return tensor->size;
}

template<>
auto THTensor<real>::strides() const -> long_range {
  return long_range(tensor->stride, tensor->stride + tensor->nDimension);
}

template<>
const long* THTensor<real>::rawStrides() const {
  return tensor->stride;
}

template<>
std::size_t THTensor<real>::storageOffset() const {
  return tensor->storageOffset;
}

template<>
std::size_t THTensor<real>::elementSize() const {
  return sizeof(real);
}

template<>
long long THTensor<real>::numel() const {
  return THTensor_(numel)(tensor);
}

template<>
bool THTensor<real>::isContiguous() const {
  return THTensor_(isContiguous)(tensor);
}

template<>
void* THTensor<real>::data() {
  return THTensor_(data)(tensor);
}

template<>
const void* THTensor<real>::data() const {
  return THTensor_(data)(tensor);
}

template<>
auto THTensor<real>::resize(const std::initializer_list<long> &new_size) -> THTensor& {
  return resize(new_size.begin(), new_size.end());
}

template<>
auto THTensor<real>::resize(const std::vector<long> &new_size) -> THTensor& {
  return resize(new_size.begin(), new_size.end());
}

template<>
template<typename iterator>
auto THTensor<real>::resize(const iterator& begin, const iterator& end) -> THTensor& {
  THLongStorage *sizes = THLongStorage_newWithSize(std::distance(begin, end));
  long *sizes_d = sizes->data;
  for (auto it = begin; it != end; ++it)
    *sizes_d++ = *it;
  THTensor_(resize)(tensor, sizes, nullptr);
  return *this;
}

template<>
auto THTensor<real>::fill(scalar_type value) -> THTensor& {
  THTensor_(fill)(tensor, value);
  return *this;
}

template<>
auto THTensor<real>::retain() -> THTensor& {
  THTensor_(retain)(tensor);
  return *this;
}

template<>
auto THTensor<real>::free() -> THTensor& {
  THTensor_(free)(tensor);
  return *this;
}

#define non_const_cast(tensor) const_cast<THTensor&>(dynamic_cast<const THTensor&>(tensor))

template<>
auto THTensor<real>::add(const Tensor &source, scalar_type value) -> THTensor& {
  THTensor &source_t = non_const_cast(source);
  THTensor_(add)(tensor, source_t.tensor, value);
  return *this;
}

template<>
thd::Type THTensor<real>::type() const {
  return thd::type_traits<real>::type;
}

#endif

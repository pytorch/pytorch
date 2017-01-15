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
auto THTensor<real>::clone_shallow() -> THTensor* {
  THTensor_(retain)(tensor);
  return new THTensor(tensor);
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
auto THTensor<real>::resize(THLongStorage *size,
                            THLongStorage *stride) -> THTensor& {
  THTensor_(resize)(tensor, size, stride);
  return *this;
}

template<>
auto THTensor<real>::resizeAs(const Tensor& src) -> THTensor& {
  THTensor_(resizeAs)(tensor, dynamic_cast<const THTensor<real>&>(src).tensor);
  return *this;
}

template<>
template<typename iterator>
auto THTensor<real>::resize(const iterator& begin, const iterator& end) -> THTensor& {
  THLongStorage *sizes = THLongStorage_newWithSize(std::distance(begin, end));
  long *sizes_d = sizes->data;
  for (auto it = begin; it != end; ++it)
    *sizes_d++ = *it;
  // TODO this might leak on error
  THTensor_(resize)(tensor, sizes, nullptr);
  THLongStorage_free(sizes);
  return *this;
}

template<>
auto THTensor<real>::set(const Tensor& src) -> THTensor& {
  THTensor_(set)(
    tensor,
    (dynamic_cast<const THTensor<real>&>(src)).tensor
  );
  return *this;
}

template<>
auto THTensor<real>::setStorage(const Storage& storage,
                                ptrdiff_t storageOffset,
                                THLongStorage *size,
                                THLongStorage *stride) -> THTensor& {
  THTensor_(setStorage)(
    tensor,
    (dynamic_cast<const THStorage<real>&>(storage)).getRaw(),
    storageOffset,
    size,
    stride
  );
  return *this;
}

template<>
auto THTensor<real>::narrow(const Tensor& src,
                            int dimension,
                            long firstIndex,
                            long size) -> THTensor& {
  THTensor_(narrow)(
    tensor,
    (dynamic_cast<const THTensor<real>&>(src)).tensor,
    dimension,
    firstIndex,
    size
  );
  return *this;
}

template<>
auto THTensor<real>::select(const Tensor& src, int dimension,
                            long sliceIndex) -> THTensor& {
  THTensor_(select)(
    tensor,
    (dynamic_cast<const THTensor<real>&>(src)).tensor,
    dimension,
    sliceIndex
  );
  return *this;
}

template<>
auto THTensor<real>::transpose(const Tensor& src, int dimension1,
                               int dimension2) -> THTensor& {
  auto src_raw = (dynamic_cast<const THTensor<real>&>(src)).tensor;
  if (tensor != src_raw)
    set(src);
  THTensor_(transpose)(tensor, src_raw, dimension1, dimension2);
  return *this;
}

template<>
auto THTensor<real>::unfold(const Tensor& src, int dimension,
                            long size, long step) ->THTensor& {
  auto src_raw = (dynamic_cast<const THTensor<real>&>(src)).tensor;
  if (tensor != src_raw)
    set(src);
  THTensor_(unfold)(tensor, src_raw, dimension, size, step);
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

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "tensors/generic/THTensor.cpp"
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
auto THTensor<real>::contiguous() const -> std::unique_ptr<Tensor> {
  return std::unique_ptr<Tensor>(new THTensor(THTensor_(newContiguous)(tensor)));
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
void* THTensor<real>::cdata() {
  return tensor;
}

template<>
const void* THTensor<real>::cdata() const {
  return tensor;
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
                                const long_range& size,
                                const long_range& stride) -> THTensor& {
  auto raw_storage = dynamic_cast<const THStorage<real>&>(storage).getRaw();
  int nDimension = size.size();
  auto raw_size = const_cast<long*>(size.data());
  auto raw_stride = const_cast<long*>(stride.empty() ? nullptr : stride.data());
  THTensor_(setStorageNd)(
      tensor, raw_storage, storageOffset, nDimension, raw_size, raw_stride);
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
                            long size, long step) -> THTensor& {
  auto src_raw = (dynamic_cast<const THTensor<real>&>(src)).tensor;
  if (tensor != src_raw)
    set(src);
  THTensor_(unfold)(tensor, src_raw, dimension, size, step);
  return *this;
}

template<>
auto THTensor<real>::squeeze(const Tensor& src, int dimension) -> THTensor& {
  auto src_raw = (dynamic_cast<const THTensor<real>&>(src)).tensor;
  THTensor_(squeeze1d)(tensor, src_raw, dimension);
  return *this;
}

template<>
auto THTensor<real>::unsqueeze(const Tensor& src, int dimension) -> THTensor& {
  auto src_raw = (dynamic_cast<const THTensor<real>&>(src)).tensor;
  THTensor_(unsqueeze1d)(tensor, src_raw, dimension);
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
#define non_const_long_cast(tensor) const_cast<THTensor<long>&>(dynamic_cast<const THTensor<long>&>(tensor))

template<>
auto THTensor<real>::cat(const std::vector<Tensor*>& src, int dimension) -> THTensor& {
  int num_inputs = src.size();
  std::vector<tensor_type*> inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs[i] = non_const_cast(*src[i]).tensor;
  }
  THTensor_(catArray)(tensor, inputs.data(), num_inputs, dimension);
  return *this;
}

template<>
auto THTensor<real>::gather(const Tensor& src, int dimension, const Tensor& index) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor<long> &index_t = non_const_long_cast(index);
  THTensor_(gather)(tensor, src_t.tensor, dimension, index_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::scatter(int dimension, const Tensor& index, const Tensor& src) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor<long> &index_t = non_const_long_cast(index);
  THTensor_(scatter)(tensor, dimension, index_t.tensor, src_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::scatterFill(int dimension, const Tensor& index, scalar_type value) -> THTensor& {
  THTensor<long> &index_t = non_const_long_cast(index);
  THTensor_(scatterFill)(tensor, dimension, index_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::dot(const Tensor &src) -> scalar_type {
  THTensor &src_t = non_const_cast(src);
  return THTensor_(dot)(tensor, src_t.tensor);
}

template<>
auto THTensor<real>::minall() -> scalar_type {
  return THTensor_(minall)(tensor);
}

template<>
auto THTensor<real>::maxall() -> scalar_type {
  return THTensor_(maxall)(tensor);
}

template<>
auto THTensor<real>::sumall() -> scalar_type {
  return THTensor_(sumall)(tensor);
}

template<>
auto THTensor<real>::prodall() -> scalar_type {
  return THTensor_(prodall)(tensor);
}

template<>
auto THTensor<real>::neg(const Tensor &src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor &src_t = non_const_cast(src);
  THTensor_(neg)(tensor, src_t.tensor);
#else
  throw std::runtime_error("neg is only available for `float` and `double` types");
#endif // defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return *this;
}

template<>
auto THTensor<real>::cinv(const Tensor &src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor &src_t = non_const_cast(src);
  THTensor_(cinv)(tensor, src_t.tensor);
#else
  throw std::runtime_error("cinv is only available for `float` and `double` types");
#endif // defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return *this;
}

template<>
auto THTensor<real>::add(const Tensor &src, scalar_type value) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(add)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::sub(const Tensor &src, scalar_type value) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(sub)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::mul(const Tensor &src, scalar_type value) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(mul)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::div(const Tensor &src, scalar_type value) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(div)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::fmod(const Tensor &src, scalar_type value) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(fmod)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::remainder(const Tensor &src, scalar_type value) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(remainder)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::clamp(const Tensor &src, scalar_type min_value, scalar_type max_value) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(clamp)(tensor, src_t.tensor, min_value, max_value);
  return *this;
}

template<>
auto THTensor<real>::cadd(const Tensor& src1, scalar_type value, const Tensor& src2) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);

  THSTensor<real>* src2_sparse;
  if ((src2_sparse = dynamic_cast<THSTensor<real>*>(const_cast<Tensor*>(&src2)))) {
    THSTensor_(spcadd)(tensor, src1_t.tensor, value, src2_sparse->tensor);
    return *this;
  } else {
    THTensor &src2_t = non_const_cast(src2);
    THTensor_(cadd)(tensor, src1_t.tensor, value, src2_t.tensor);
  }
  return *this;
}

template<>
auto THTensor<real>::cadd(const Tensor& src1, const Tensor& src2) -> THTensor& {
  return cadd(src1, static_cast<scalar_type>(1), src2);
}

template<>
auto THTensor<real>::csub(const Tensor& src1, scalar_type value, const Tensor& src2) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);
  THTensor &src2_t = non_const_cast(src2);
  THTensor_(csub)(tensor, src1_t.tensor, value, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cmul(const Tensor& src1, const Tensor& src2) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);
  THTensor &src2_t = non_const_cast(src2);
  THTensor_(cmul)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cpow(const Tensor& src1, const Tensor& src2) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);
  THTensor &src2_t = non_const_cast(src2);
  THTensor_(cpow)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cdiv(const Tensor& src1, const Tensor& src2) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);
  THTensor &src2_t = non_const_cast(src2);
  THTensor_(cdiv)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cfmod(const Tensor& src1, const Tensor& src2) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);
  THTensor &src2_t = non_const_cast(src2);
  THTensor_(cfmod)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cremainder(const Tensor& src1, const Tensor& src2) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);
  THTensor &src2_t = non_const_cast(src2);
  THTensor_(cremainder)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addcmul(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);
  THTensor &src2_t = non_const_cast(src2);
  THTensor &src3_t = non_const_cast(src3);
  THTensor_(addcmul)(tensor, src1_t.tensor, value, src2_t.tensor, src3_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addcdiv(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);
  THTensor &src2_t = non_const_cast(src2);
  THTensor &src3_t = non_const_cast(src3);
  THTensor_(addcdiv)(tensor, src1_t.tensor, value, src2_t.tensor, src3_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addmv(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat, const Tensor& vec) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor &mat_t = non_const_cast(mat);
  THTensor &vec_t = non_const_cast(vec);
  THTensor_(addmv)(tensor, beta, src_t.tensor, alpha, mat_t.tensor, vec_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat1, const Tensor& mat2) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor &mat1_t = non_const_cast(mat1);
  THTensor &mat2_t = non_const_cast(mat2);
  THTensor_(addmm)(tensor, beta, src_t.tensor, alpha, mat1_t.tensor, mat2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addr(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& vec1, const Tensor& vec2) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor &vec1_t = non_const_cast(vec1);
  THTensor &vec2_t = non_const_cast(vec2);
  THTensor_(addr)(tensor, beta, src_t.tensor, alpha, vec1_t.tensor, vec2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor &batch1_t = non_const_cast(batch1);
  THTensor &batch2_t = non_const_cast(batch2);
  THTensor_(addbmm)(tensor, beta, src_t.tensor, alpha, batch1_t.tensor, batch2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::baddbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor &batch1_t = non_const_cast(batch1);
  THTensor &batch2_t = non_const_cast(batch2);
  THTensor_(baddbmm)(tensor, beta, src_t.tensor, alpha, batch1_t.tensor, batch2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::match(const Tensor& m1, const Tensor& m2, scalar_type gain) -> THTensor& {
  THTensor &m1_t = non_const_cast(m1);
  THTensor &m2_t = non_const_cast(m2);
  THTensor_(match)(tensor, m1_t.tensor, m2_t.tensor, gain);
  return *this;
}

template<>
auto THTensor<real>::max(const Tensor& indices_, const Tensor& src, int dimension) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor<long> &indices__t = non_const_long_cast(indices_);
  THTensor_(max)(tensor, indices__t.tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::min(const Tensor& indices_, const Tensor& src, int dimension) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor<long> &indices__t = non_const_long_cast(indices_);
  THTensor_(min)(tensor, indices__t.tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor<long> &indices__t = non_const_long_cast(indices_);
  THTensor_(kthvalue)(tensor, indices__t.tensor, src_t.tensor, k, dimension);
  return *this;
}

template<>
auto THTensor<real>::mode(const Tensor& indices_, const Tensor& src, int dimension) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor<long> &indices__t = non_const_long_cast(indices_);
  THTensor_(mode)(tensor, indices__t.tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::median(const Tensor& indices_, const Tensor& src, int dimension) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor<long> &indices__t = non_const_long_cast(indices_);
  THTensor_(median)(tensor, indices__t.tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::sum(const Tensor& src, int dimension) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(sum)(tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::prod(const Tensor& src, int dimension) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(prod)(tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::cumsum(const Tensor& src, int dimension) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(cumsum)(tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::cumprod(const Tensor& src, int dimension) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(cumprod)(tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::sign(const Tensor& src) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(sign)(tensor, src_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::trace() -> scalar_type {
  return THTensor_(trace)(tensor);
}

template<>
auto THTensor<real>::cross(const Tensor& src1, const Tensor& src2, int dimension) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);
  THTensor &src2_t = non_const_cast(src2);
  THTensor_(cross)(tensor, src1_t.tensor, src2_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::cmax(const Tensor& src1, const Tensor& src2) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);
  THTensor &src2_t = non_const_cast(src2);
  THTensor_(cmax)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cmin(const Tensor& src1, const Tensor& src2) -> THTensor& {
  THTensor &src1_t = non_const_cast(src1);
  THTensor &src2_t = non_const_cast(src2);
  THTensor_(cmin)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cmaxValue(const Tensor& src, scalar_type value) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(cmaxValue)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::cminValue(const Tensor& src, scalar_type value) -> THTensor& {
  THTensor &src_t = non_const_cast(src);
  THTensor_(cminValue)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::zero() -> THTensor& {
  THTensor_(zero)(tensor);
  return *this;
}

template<>
thpp::Type THTensor<real>::type() const {
  return thpp::type_traits<real>::type;
}

template<>
bool THTensor<real>::isCuda() const {
  return false;
}

template<>
bool THTensor<real>::isSparse() const {
  return false;
}

template<>
int THTensor<real>::getDevice() const {
  return -1;
}

template<>
std::unique_ptr<Tensor> THTensor<real>::newTensor() const {
  return std::unique_ptr<Tensor>(new THTensor<real>());
}

#endif

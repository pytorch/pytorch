#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "tensors/generic/THCTensor.cpp"
#else

template<>
THCTensor<real>::THCTensor(THCState* state):
  tensor(THCTensor_(new)(state)), state(state)
  {};

template<>
THCTensor<real>::THCTensor(THCState* state, THCRealTensor *wrapped):
  tensor(wrapped), state(state)
  {};

template<>
THCTensor<real>::~THCTensor() {
  if (tensor)
    THCTensor_(free)(state, tensor);
}

template<>
auto THCTensor<real>::clone() const -> THCTensor* {
  return new THCTensor(state, THCTensor_(newClone)(state, tensor));
}

template<>
auto THCTensor<real>::clone_shallow() -> THCTensor* {
  THCTensor_(retain)(state, tensor);
  return new THCTensor(state, tensor);
}

template<>
auto THCTensor<real>::contiguous() const -> std::unique_ptr<Tensor> {
  return std::unique_ptr<Tensor>(
      new THCTensor(state, THCTensor_(newContiguous)(state, tensor)));
}

template<>
int THCTensor<real>::nDim() const {
  return tensor->nDimension;
}

template<>
auto THCTensor<real>::sizes() const -> long_range {
  return std::vector<long>(tensor->size, tensor->size + tensor->nDimension);
}

template<>
const long* THCTensor<real>::rawSizes() const {
  return tensor->size;
}

template<>
auto THCTensor<real>::strides() const -> long_range {
  return long_range(tensor->stride, tensor->stride + tensor->nDimension);
}

template<>
const long* THCTensor<real>::rawStrides() const {
  return tensor->stride;
}

template<>
std::size_t THCTensor<real>::storageOffset() const {
  return tensor->storageOffset;
}

template<>
std::size_t THCTensor<real>::elementSize() const {
  return sizeof(real);
}

template<>
long long THCTensor<real>::numel() const {
  return THCTensor_(numel)(state, tensor);
}

template<>
bool THCTensor<real>::isContiguous() const {
  return THCTensor_(isContiguous)(state, tensor);
}

template<>
void* THCTensor<real>::data() {
  return THCTensor_(data)(state, tensor);
}

template<>
const void* THCTensor<real>::data() const {
  return THCTensor_(data)(state, tensor);
}

template<>
void* THCTensor<real>::cdata() {
  return tensor;
}

template<>
const void* THCTensor<real>::cdata() const {
  return tensor;
}

template<>
auto THCTensor<real>::resize(const std::initializer_list<long> &new_size) -> THCTensor& {
  return resize(new_size.begin(), new_size.end());
}

template<>
auto THCTensor<real>::resize(const std::vector<long> &new_size) -> THCTensor& {
  return resize(new_size.begin(), new_size.end());
}

template<>
auto THCTensor<real>::resize(THLongStorage *size,
                             THLongStorage *stride) -> THCTensor& {
  THCTensor_(resize)(state, tensor, size, stride);
  return *this;
}

template<>
auto THCTensor<real>::resizeAs(const Tensor& src) -> THCTensor& {
  THCTensor_(resizeAs)(state, tensor, dynamic_cast<const THCTensor<real>&>(src).tensor);
  return *this;
}

template<>
template<typename iterator>
auto THCTensor<real>::resize(const iterator& begin, const iterator& end) -> THCTensor& {
  THLongStorage *sizes = THLongStorage_newWithSize(std::distance(begin, end));
  long *sizes_d = sizes->data;
  for (auto it = begin; it != end; ++it)
    *sizes_d++ = *it;
  // TODO this might leak on error
  THCTensor_(resize)(state, tensor, sizes, nullptr);
  THLongStorage_free(sizes);
  return *this;
}

template<>
auto THCTensor<real>::set(const Tensor& src) -> THCTensor& {
  THCTensor_(set)(state,
    tensor,
    (dynamic_cast<const THCTensor<real>&>(src)).tensor
  );
  return *this;
}

template<>
auto THCTensor<real>::setStorage(const Storage& storage,
                                 ptrdiff_t storageOffset,
                                 const long_range& size,
                                 const long_range& stride) -> THCTensor& {
  auto raw_storage = dynamic_cast<const THCStorage<real>&>(storage).getRaw();
  int nDimension = size.size();
  auto raw_size = const_cast<long*>(size.data());
  auto raw_stride = const_cast<long*>(stride.empty() ? nullptr : stride.data());
  THCTensor_(setStorageNd)(
      state, tensor, raw_storage, storageOffset, nDimension, raw_size, raw_stride);
  return *this;
}

template<>
auto THCTensor<real>::setStorage(const Storage& storage,
                                 ptrdiff_t storageOffset,
                                 THLongStorage *size,
                                 THLongStorage *stride) -> THCTensor& {
  THCTensor_(setStorage)(state,
    tensor,
    (dynamic_cast<const THCStorage<real>&>(storage)).getRaw(),
    storageOffset,
    size,
    stride
  );
  return *this;
}

template<>
auto THCTensor<real>::narrow(const Tensor& src,
                            int dimension,
                            long firstIndex,
                            long size) -> THCTensor& {
  THCTensor_(narrow)(state,
    tensor,
    (dynamic_cast<const THCTensor<real>&>(src)).tensor,
    dimension,
    firstIndex,
    size
  );
  return *this;
}

template<>
auto THCTensor<real>::select(const Tensor& src, int dimension,
                            long sliceIndex) -> THCTensor& {
  THCTensor_(select)(state,
    tensor,
    (dynamic_cast<const THCTensor<real>&>(src)).tensor,
    dimension,
    sliceIndex
  );
  return *this;
}

template<>
auto THCTensor<real>::transpose(const Tensor& src, int dimension1,
                               int dimension2) -> THCTensor& {
  auto src_raw = (dynamic_cast<const THCTensor<real>&>(src)).tensor;
  if (tensor != src_raw)
    set(src);
  THCTensor_(transpose)(state, tensor, src_raw, dimension1, dimension2);
  return *this;
}

template<>
auto THCTensor<real>::unfold(const Tensor& src, int dimension,
                            long size, long step) ->THCTensor& {
  auto src_raw = (dynamic_cast<const THCTensor<real>&>(src)).tensor;
  if (tensor != src_raw)
    set(src);
  THCTensor_(unfold)(state, tensor, src_raw, dimension, size, step);
  return *this;
}

template<>
auto THCTensor<real>::squeeze(const Tensor& src, int dimension) -> THCTensor& {
  auto src_raw = (dynamic_cast<const THCTensor<real>&>(src)).tensor;
  THCTensor_(squeeze1d)(state, tensor, src_raw, dimension);
  return *this;
}

template<>
auto THCTensor<real>::unsqueeze(const Tensor& src, int dimension) -> THCTensor& {
  auto src_raw = (dynamic_cast<const THCTensor<real>&>(src)).tensor;
  THCTensor_(unsqueeze1d)(state, tensor, src_raw, dimension);
  return *this;
}

#ifdef THC_REAL_IS_HALF
#define cast_scalar(v) THC_float2half(v)
#define uncast_scalar(v) THC_half2float(v)
#else
#define cast_scalar(v) v
#define uncast_scalar(v) v
#endif

template<>
auto THCTensor<real>::fill(scalar_type value) -> THCTensor& {
  THCTensor_(fill)(state, tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::retain() -> THCTensor& {
  THCTensor_(retain)(state, tensor);
  return *this;
}

template<>
auto THCTensor<real>::free() -> THCTensor& {
  THCTensor_(free)(state, tensor);
  return *this;
}

#define non_const_cast(tensor) const_cast<THCTensor&>(dynamic_cast<const THCTensor&>(tensor))
#define non_const_long_cast(tensor) const_cast<THCTensor<long>&>(dynamic_cast<const THCTensor<long>&>(tensor))

template<>
auto THCTensor<real>::cat(const std::vector<Tensor*>& src, int dimension) -> THCTensor& {
  int num_inputs = src.size();
  std::vector<tensor_type*> inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs[i] = non_const_cast(*src[i]).tensor;
  }
  THCTensor_(catArray)(state, tensor, inputs.data(), num_inputs, dimension);
  return *this;
}

template<>
auto THCTensor<real>::gather(const Tensor& src, int dimension, const Tensor& index) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor<long> &index_t = non_const_long_cast(index);
  THCTensor_(gather)(state, tensor, src_t.tensor, dimension, index_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::scatter(int dimension, const Tensor& index, const Tensor& src) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor<long> &index_t = non_const_long_cast(index);
  THCTensor_(scatter)(state, tensor, dimension, index_t.tensor, src_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::scatterFill(int dimension, const Tensor& index, scalar_type value) -> THCTensor& {
  THCTensor<long> &index_t = non_const_long_cast(index);
  THCTensor_(scatterFill)(state, tensor, dimension, index_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::dot(const Tensor &src) -> scalar_type {
  THCTensor &src_t = non_const_cast(src);
  return THCTensor_(dot)(state, tensor, src_t.tensor);
}

template<>
auto THCTensor<real>::minall() -> scalar_type {
  return uncast_scalar(THCTensor_(minall)(state, tensor));
}

template<>
auto THCTensor<real>::maxall() -> scalar_type {
  return uncast_scalar(THCTensor_(maxall)(state, tensor));
}

template<>
auto THCTensor<real>::sumall() -> scalar_type {
  return THCTensor_(sumall)(state, tensor);
}

template<>
auto THCTensor<real>::prodall() -> scalar_type {
  return THCTensor_(prodall)(state, tensor);
}

template<>
auto THCTensor<real>::neg(const Tensor &src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(neg)(state, tensor, src_t.tensor);
#else
  throw std::runtime_error("neg is only available for `float` and `double` types");
#endif // defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return *this;
}

template<>
auto THCTensor<real>::cinv(const Tensor &src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(cinv)(state, tensor, src_t.tensor);
#else
  throw std::runtime_error("cinv is only available for `float` and `double` types");
#endif // defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return *this;
}

template<>
auto THCTensor<real>::add(const Tensor &src, scalar_type value) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(add)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::sub(const Tensor &src, scalar_type value) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(sub)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::mul(const Tensor &src, scalar_type value) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(mul)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::div(const Tensor &src, scalar_type value) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(div)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::fmod(const Tensor &src, scalar_type value) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(fmod)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::remainder(const Tensor &src, scalar_type value) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(remainder)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::clamp(const Tensor &src, scalar_type min_value, scalar_type max_value) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(clamp)(state, tensor, src_t.tensor, cast_scalar(min_value), cast_scalar(max_value));
  return *this;
}

template<>
auto THCTensor<real>::cadd(const Tensor& src1, scalar_type value, const Tensor& src2) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor_(cadd)(state, tensor, src1_t.tensor, cast_scalar(value), src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cadd(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  return cadd(src1, static_cast<scalar_type>(1), src2);
}

template<>
auto THCTensor<real>::csub(const Tensor& src1, scalar_type value, const Tensor& src2) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor_(csub)(state, tensor, src1_t.tensor, cast_scalar(value), src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cmul(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor_(cmul)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cpow(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor_(cpow)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cdiv(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor_(cdiv)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cfmod(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor_(cfmod)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cremainder(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor_(cremainder)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addcmul(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor &src3_t = non_const_cast(src3);
  THCTensor_(addcmul)(state, tensor, src1_t.tensor, cast_scalar(value), src2_t.tensor, src3_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addcdiv(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor &src3_t = non_const_cast(src3);
  THCTensor_(addcdiv)(state, tensor, src1_t.tensor, cast_scalar(value), src2_t.tensor, src3_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addmv(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat, const Tensor& vec) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor &mat_t = non_const_cast(mat);
  THCTensor &vec_t = non_const_cast(vec);
  THCTensor_(addmv)(state, tensor, cast_scalar(beta), src_t.tensor, cast_scalar(alpha), mat_t.tensor, vec_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat1, const Tensor& mat2) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor &mat1_t = non_const_cast(mat1);
  THCTensor &mat2_t = non_const_cast(mat2);
  THCTensor_(addmm)(state, tensor, cast_scalar(beta), src_t.tensor, cast_scalar(alpha), mat1_t.tensor, mat2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addr(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& vec1, const Tensor& vec2) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor &vec1_t = non_const_cast(vec1);
  THCTensor &vec2_t = non_const_cast(vec2);
  THCTensor_(addr)(state, tensor, cast_scalar(beta), src_t.tensor, cast_scalar(alpha), vec1_t.tensor, vec2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor &batch1_t = non_const_cast(batch1);
  THCTensor &batch2_t = non_const_cast(batch2);
  THCTensor_(addbmm)(state, tensor, cast_scalar(beta), src_t.tensor, cast_scalar(alpha), batch1_t.tensor, batch2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::baddbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor &batch1_t = non_const_cast(batch1);
  THCTensor &batch2_t = non_const_cast(batch2);
  THCTensor_(baddbmm)(state, tensor, cast_scalar(beta), src_t.tensor, cast_scalar(alpha), batch1_t.tensor, batch2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::match(const Tensor& m1, const Tensor& m2, scalar_type gain) -> THCTensor& {
  throw std::runtime_error("unsupported operation 'match'");
}

template<>
auto THCTensor<real>::max(const Tensor& indices_, const Tensor& src, int dimension) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor<long> &indices__t = non_const_long_cast(indices_);
  THCTensor_(max)(state, tensor, indices__t.tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THCTensor<real>::min(const Tensor& indices_, const Tensor& src, int dimension) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor<long> &indices__t = non_const_long_cast(indices_);
  THCTensor_(min)(state, tensor, indices__t.tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THCTensor<real>::kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension) -> THCTensor& {
  throw std::runtime_error("unsupported operation 'kthvalue'");
}

template<>
auto THCTensor<real>::mode(const Tensor& indices_, const Tensor& src, int dimension) -> THCTensor& {
  throw std::runtime_error("unsupported operation 'mode'");
}

template<>
auto THCTensor<real>::median(const Tensor& indices_, const Tensor& src, int dimension) -> THCTensor& {
  throw std::runtime_error("unsupported operation 'median'");
}

template<>
auto THCTensor<real>::sum(const Tensor& src, int dimension) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(sum)(state, tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THCTensor<real>::prod(const Tensor& src, int dimension) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(prod)(state, tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THCTensor<real>::cumsum(const Tensor& src, int dimension) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(cumsum)(state, tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THCTensor<real>::cumprod(const Tensor& src, int dimension) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(cumprod)(state, tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THCTensor<real>::sign(const Tensor& src) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(sign)(state, tensor, src_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::trace() -> scalar_type {
  return THCTensor_(trace)(state, tensor);
}

template<>
auto THCTensor<real>::cross(const Tensor& src1, const Tensor& src2, int dimension) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor_(cross)(state, tensor, src1_t.tensor, src2_t.tensor, dimension);
  return *this;
}

template<>
auto THCTensor<real>::cmax(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor_(cmax)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cmin(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  THCTensor &src1_t = non_const_cast(src1);
  THCTensor &src2_t = non_const_cast(src2);
  THCTensor_(cmin)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cmaxValue(const Tensor& src, scalar_type value) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(cmaxValue)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::cminValue(const Tensor& src, scalar_type value) -> THCTensor& {
  THCTensor &src_t = non_const_cast(src);
  THCTensor_(cminValue)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::zero() -> THCTensor& {
  THCTensor_(zero)(state, tensor);
  return *this;
}

template<>
thpp::Type THCTensor<real>::type() const {
  return thpp::type_traits<real>::type;
}

template<>
bool THCTensor<real>::isCuda() const {
  return true;
}

template<>
bool THCTensor<real>::isSparse() const {
  return false;
}

template<>
int THCTensor<real>::getDevice() const {
  return tensor->storage->device;
}

template<>
std::unique_ptr<Tensor> THCTensor<real>::newTensor() const {
  return std::unique_ptr<Tensor>(new THCTensor<real>(state));
}

#undef cast_scalar
#undef uncast_scalar

#endif

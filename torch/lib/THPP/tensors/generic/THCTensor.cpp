#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "tensors/generic/THCTensor.cpp"
#else

#define const_tensor_cast(tensor) \
  dynamic_cast<const THCTensor&>(tensor)
#define const_storage_cast(storage) \
  dynamic_cast<const THCStorage<real>&>(storage)
#define const_long_cast(tensor) \
  dynamic_cast<const THCTensor<long>&>(tensor)
#define const_float_cast(tensor) \
  dynamic_cast<const THCTensor<float>&>(tensor)
#define const_double_cast(tensor) \
  dynamic_cast<const THCTensor<double>&>(tensor)
#define const_byte_cast(tensor) \
  dynamic_cast<const THCTensor<unsigned char>&>(tensor)

#ifdef THC_REAL_IS_HALF
#define cast_scalar(v) THC_float2half(v)
#define uncast_scalar(v) THC_half2float(v)
#else
#define cast_scalar(v) v
#define uncast_scalar(v) v
#endif

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
auto THCTensor<real>::newSelect(int dimension, long sliceIndex) const -> THCTensor* {
  throw std::runtime_error("newSelect is not yet available for CUDA tensors");
}

template<>
auto THCTensor<real>::newNarrow(int dimension, long firstIndex, long size) const -> THCTensor* {
  return new THCTensor(state, THCTensor_(newNarrow)(state, tensor, dimension, firstIndex, size));
}

template<>
auto THCTensor<real>::newTranspose(int dimension1, int dimension2) const -> THCTensor* {
  return new THCTensor(state, THCTensor_(newTranspose)(state, tensor, dimension1, dimension2));
}

template<>
auto THCTensor<real>::newUnfold(int dimension, long size, long step) const -> THCTensor* {
  throw std::runtime_error("newUnfold is not yet available for CUDA tensors");
}

template<>
auto THCTensor<real>::newExpand(const long_range& size) const -> THCTensor* {
  THLongStorage *size_storage = THLongStorage_newWithSize(size.size());
  std::memcpy(size_storage->data, size.data(), sizeof(long) * size.size());
  // TODO this might leak on error
  auto expanded = new THCTensor(state, THCTensor_(newExpand)(state, tensor, size_storage));
  THLongStorage_free(size_storage);
  return expanded;
}

template<>
auto THCTensor<real>::newView(const long_range& size) const -> THCTensor* {
  THLongStorage *size_storage = THLongStorage_newWithSize(size.size());
  std::memcpy(size_storage->data, size.data(), sizeof(long) * size.size());
  // TODO this might leak on error
  auto viewed = new THCTensor(state, THCTensor_(newView)(state, tensor, size_storage));
  THLongStorage_free(size_storage);
  return viewed;
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
auto THCTensor<real>::resize(const std::initializer_list<long> &new_size)
    -> THCTensor& {
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
  THCTensor_(resizeAs)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
}

template<>
template<typename iterator>
auto THCTensor<real>::resize(const iterator& begin,
                             const iterator& end) -> THCTensor& {
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
    const_tensor_cast(src).tensor
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
    const_storage_cast(storage).getRaw(),
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
    const_tensor_cast(src).tensor,
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
    const_tensor_cast(src).tensor,
    dimension,
    sliceIndex
  );
  return *this;
}

template<>
auto THCTensor<real>::transpose(const Tensor& src, int dimension1,
                               int dimension2) -> THCTensor& {
  auto src_raw = const_tensor_cast(src).tensor;
  if (tensor != src_raw)
    set(src);
  THCTensor_(transpose)(state, tensor, src_raw, dimension1, dimension2);
  return *this;
}

template<>
auto THCTensor<real>::unfold(const Tensor& src, int dimension,
                            long size, long step) ->THCTensor& {
  auto src_raw = const_tensor_cast(src).tensor;
  if (tensor != src_raw)
    set(src);
  THCTensor_(unfold)(state, tensor, src_raw, dimension, size, step);
  return *this;
}

template<>
auto THCTensor<real>::squeeze(const Tensor& src) -> THCTensor& {
  auto src_raw = dynamic_cast<const THCTensor<real>&>(src).tensor;
  THCTensor_(squeeze)(state, tensor, src_raw);
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

#define LAPACK_ERROR "Lapack operations not available for CUDA tensors"

template<>
auto THCTensor<real>::gesv(const Tensor& ra, const Tensor& b, const Tensor& a) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
  return *this;
}

template<>
auto THCTensor<real>::trtrs(const Tensor& ra, const Tensor& b, const Tensor& a,
                            const char *uplo, const char *trans, const char *diag) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::gels(const Tensor& ra, const Tensor& b, const Tensor& a) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::syev(const Tensor& rv, const Tensor& a,
                           const char *jobz, const char *uplo) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::geev(const Tensor& rv, const Tensor& a, const char *jobvr) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::gesvd(const Tensor& rs, const Tensor& rv,
                            const Tensor& a, const char *jobu) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::gesvd2(const Tensor& rs, const Tensor& rv, const Tensor& ra,
                             const Tensor& a, const char *jobu) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::getri(const Tensor& a) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::potrf(const Tensor& a, const char *uplo) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::potrs(const Tensor& b, const Tensor& a, const char *uplo) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::potri(const Tensor& a, const char *uplo) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::qr(const Tensor& rr, const Tensor& a) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::geqrf(const Tensor& rtau, const Tensor& a) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::orgqr(const Tensor& a, const Tensor& tau) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::ormqr(const Tensor& a, const Tensor& tau, const Tensor& c,
                            const char *side, const char *trans) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::pstrf(const Tensor& rpiv, const Tensor& a,
                            const char *uplo, scalar_type tol) -> THCTensor& {
  throw std::runtime_error(LAPACK_ERROR);
}

template<>
auto THCTensor<real>::fill(scalar_type value) -> THCTensor& {
  THCTensor_(fill)(state, tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::maskedFill(const Tensor& mask, scalar_type value) -> THCTensor& {
  THCTensor_(maskedFill)(state, tensor, const_byte_cast(mask).tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::maskedCopy(const Tensor& mask, const Tensor& src) -> THCTensor& {
  THCTensor_(maskedCopy)(state, tensor, const_byte_cast(mask).tensor,
                         const_tensor_cast(src).tensor);
  return *this;
}

template<>
auto THCTensor<real>::maskedSelect(const Tensor& src, const Tensor& mask) -> THCTensor& {
  THCTensor_(maskedSelect)(state, tensor, const_tensor_cast(src).tensor,
                           const_byte_cast(mask).tensor);
  return *this;
}

template<>
auto THCTensor<real>::nonzero(const Tensor& subscript) -> THCTensor& {
  THCTensor_(nonzero)(state, const_long_cast(subscript).tensor, tensor);
  return *this;
}

template<>
auto THCTensor<real>::indexSelect(const Tensor& src, int dim, const Tensor& index) -> THCTensor& {
  THCTensor_(indexSelect)(state, tensor, const_tensor_cast(src).tensor, dim,
                          const_long_cast(index).tensor);
  return *this;
}

template<>
auto THCTensor<real>::indexCopy(int dim, const Tensor& index, const Tensor& src) -> THCTensor& {
  THCTensor_(indexCopy)(state, tensor, dim, const_long_cast(index).tensor,
                        const_tensor_cast(src).tensor);
  return *this;
}

template<>
auto THCTensor<real>::indexAdd(int dim, const Tensor& index, const Tensor& src) -> THCTensor& {
  THCTensor_(indexAdd)(state, tensor, dim, const_long_cast(index).tensor,
                       const_tensor_cast(src).tensor);
  return *this;
}

template<>
auto THCTensor<real>::indexFill(int dim, const Tensor& index, scalar_type value) -> THCTensor& {
  THCTensor_(indexFill)(state, tensor, dim, const_long_cast(index).tensor, cast_scalar(value));
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

template<>
auto THCTensor<real>::diag(const Tensor& src, int k) -> THCTensor& {
  THCTensor_(diag)(state, tensor, const_tensor_cast(src).tensor, k);
  return *this;
}

template<>
auto THCTensor<real>::eye(long n, long m) -> THCTensor& {
  throw std::runtime_error("THCTensor::eye() not implemented");
}

template<>
auto THCTensor<real>::range(scalar_type xmin, scalar_type xmax,
                            scalar_type step) -> THCTensor& {
  throw std::runtime_error("THCTensor::range() not implemented");
}

template<>
auto THCTensor<real>::randperm(const Generator& _generator, long n) -> THCTensor& {
  throw std::runtime_error("THCTensor::randperm() not implemented");
}

template<>
auto THCTensor<real>::sort(const Tensor& ri, const Tensor& src,
                           int dimension, int desc) -> THCTensor& {
  THCTensor_(sort)(
    state,
    tensor,
    const_long_cast(ri).tensor,
    const_tensor_cast(src).tensor,
    dimension,
    desc
  );
  return *this;
}

template<>
auto THCTensor<real>::topk(const Tensor& ri, const Tensor& src,
                           long k, int dim, int dir, int sorted) -> THCTensor& {
#ifdef THC_REAL_IS_FLOAT
  THCTensor_(topk)(
    state,
    tensor,
    const_long_cast(ri).tensor,
    const_tensor_cast(src).tensor,
    k,
    dim,
    dir,
    sorted
  );
  return *this;
#else
  throw std::runtime_error("THCTensor::topk() is implemented only for float type");
#endif
}

template<>
auto THCTensor<real>::tril(const Tensor& src, long k) -> THCTensor& {
  THCTensor_(tril)(state, tensor, const_tensor_cast(src).tensor, k);
  return *this;
}

template<>
auto THCTensor<real>::triu(const Tensor& src, long k) -> THCTensor& {
  THCTensor_(triu)(state, tensor, const_tensor_cast(src).tensor, k);
  return *this;
}

template<>
auto THCTensor<real>::catArray(const std::vector<Tensor*>& inputs_vec,
                              int dimension) -> THCTensor& {
  int numInputs = inputs_vec.size();
  tensor_type *inputs[numInputs];
  for (std::size_t i = 0; i < numInputs; i++)
    inputs[i] = const_tensor_cast(*inputs_vec[i]).tensor;
  THCTensor_(catArray)(state, tensor, inputs, numInputs, dimension);
  return *this;
}

template<>
int THCTensor<real>::equal(const Tensor& other) const {
  return THCTensor_(equal)(state, tensor, const_tensor_cast(other).tensor);
}

  // Note: the order in *Value and *Tensor is reversed compared to
  // the declarations in TH/generic/THCTensorMath.h, so for instance
  // the call THRealTensor_ltTensor(r, ta, tb) is equivalent to
  // ta->ltTensor(r, tb). It is done this way so that the first
  // argument can be casted onto a byte tensor type

#define TENSOR_IMPLEMENT_LOGICAL(NAME)                               \
  template<>                                                         \
  auto THCTensor<real>::NAME##Value(const Tensor& r,                 \
                                    scalar_type value) -> THCTensor& { \
    if (r.type() != Type::UCHAR)                                     \
      throw std::invalid_argument("logical operator called on non-byte tensor"); \
    THCTensor_(NAME##Value)(                                         \
      state,                                                         \
      const_byte_cast(r).tensor,                                     \
      tensor,                                                        \
      cast_scalar(value)                                             \
    );                                                               \
    return *this;                                                    \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THCTensor<real>::NAME##ValueT(const Tensor& t,                \
                                     scalar_type value) -> THCTensor& { \
    THCTensor_(NAME##ValueT)(                                        \
      state,                                                         \
      tensor,                                                        \
      const_tensor_cast(t).tensor,                                   \
      cast_scalar(value)                                             \
    );                                                               \
    return *this;                                                    \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THCTensor<real>::NAME##Tensor(const Tensor& r,                \
                                     const Tensor& tb) -> THCTensor& { \
    if (r.type() != Type::UCHAR)                                     \
      throw std::invalid_argument("logical operator called on non-byte tensor"); \
    THCTensor_(NAME##Tensor)(                                        \
      state,                                                         \
      const_byte_cast(r).tensor,                                     \
      tensor,                                                        \
      const_tensor_cast(tb).tensor                                   \
    );                                                               \
    return *this;                                                    \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THCTensor<real>::NAME##TensorT(const Tensor& ta,              \
                                      const Tensor& tb) -> THCTensor& { \
    THCTensor_(NAME##TensorT)(                                       \
      state,                                                         \
      tensor,                                                        \
      const_tensor_cast(ta).tensor,                                  \
      const_tensor_cast(tb).tensor                                   \
    );                                                               \
    return *this;                                                    \
  }                                                                  \

TENSOR_IMPLEMENT_LOGICAL(lt)
TENSOR_IMPLEMENT_LOGICAL(gt)
TENSOR_IMPLEMENT_LOGICAL(le)
TENSOR_IMPLEMENT_LOGICAL(ge)
TENSOR_IMPLEMENT_LOGICAL(eq)
TENSOR_IMPLEMENT_LOGICAL(ne)

#undef TENSOR_IMPLEMENT_LOGICAL

template<>
auto THCTensor<real>::abs(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_INT) ||\
    defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THCTensor_(abs)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("absolute value is only available for\
      signed type tensors");
#endif
}

template<>
auto THCTensor<real>::sigmoid(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(sigmoid)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::log(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(log)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::log1p(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(log1p)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::exp(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(exp)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::cos(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(cos)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::acos(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(acos)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::cosh(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(cosh)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::sin(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(sin)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::asin(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(asin)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::sinh(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(sinh)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::copy(const Tensor& src) -> THCTensor& {
  // TODO: polymorphic copy
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(copy)(state, tensor, src_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::tan(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(tan)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::atan(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(atan)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::atan2(const Tensor& src1, const Tensor& src2) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(atan2)(state,
    tensor,
    const_tensor_cast(src1).tensor,
    const_tensor_cast(src2).tensor
  );
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::tanh(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(tanh)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::pow(const Tensor& src, scalar_type value) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(pow)(state, tensor, const_tensor_cast(src).tensor, value);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::tpow(scalar_type value, const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(tpow)(state, tensor, value, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::sqrt(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(sqrt)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::rsqrt(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(rsqrt)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::ceil(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(ceil)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::floor(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(floor)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::round(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(round)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::trunc(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(trunc)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::frac(const Tensor& src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor(frac)(state, tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::lerp(const Tensor& a, const Tensor& b, scalar_type weight) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(lerp)(state,
    tensor,
    const_tensor_cast(a).tensor,
    const_tensor_cast(b).tensor,
    weight
  );
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::mean(const Tensor& src, int dimension, int keepdim) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(mean)(state, tensor, const_tensor_cast(src).tensor, dimension, keepdim);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::std(const Tensor& src, int dimension, int biased, int keepdim) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(std)(state, tensor, const_tensor_cast(src).tensor, dimension, biased, keepdim);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::var(const Tensor& src, int dimension, int biased, int keepdim) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(var)(state, tensor, const_tensor_cast(src).tensor, dimension, biased, keepdim);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::norm(const Tensor& src, scalar_type value, int dimension, int keepdim) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(norm)(state, tensor, const_tensor_cast(src).tensor, value, dimension, keepdim);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::renorm(const Tensor& src,
                             scalar_type value,
                             int dimension,
                             scalar_type maxnorm) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  const THCTensor& src_t = const_tensor_cast(src);
  THCTensor_(renorm)(state, tensor, src_t.tensor, value, dimension, maxnorm);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::histc(const Tensor& src,
                            long nbins,
                            scalar_type minvalue,
                            scalar_type maxvalue) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  const THCTensor& src_t = const_tensor_cast(src);
  THCTensor_(histc)(state, tensor, src_t.tensor, nbins, minvalue, maxvalue);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::bhistc(const Tensor& src,
                             long nbins,
                             scalar_type minvalue,
                             scalar_type maxvalue) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  const THCTensor& src_t = const_tensor_cast(src);
  THCTensor_(bhistc)(state, tensor, src_t.tensor, nbins, minvalue, maxvalue);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::dist(const Tensor& src, scalar_type value) -> scalar_type {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return THCTensor_(dist)(state, tensor, const_tensor_cast(src).tensor, value);
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::meanall() -> scalar_type {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return THCTensor_(meanall)(state, tensor);
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::varall(int biased) -> scalar_type {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return THCTensor_(varall)(state, tensor, biased);
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::stdall(int biased) -> scalar_type {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return THCTensor_(stdall)(state, tensor, biased);
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::normall(scalar_type value) -> scalar_type {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return THCTensor_(normall)(state, tensor, value);
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::linspace(scalar_type a, scalar_type b, long n) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(linspace)(state, tensor, a, b, n);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::logspace(scalar_type a, scalar_type b, long n) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(logspace)(state, tensor, a, b, n);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::rand(const Generator& _generator, THLongStorage *size) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(rand)(state, tensor, size);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::randn(const Generator& _generator, THLongStorage *size) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(randn)(state, tensor, size);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
int THCTensor<real>::logicalall() {
#if defined(TH_REAL_IS_BYTE)
  return THCTensor_(logicalall)(state, tensor);
#else
  throw std::runtime_error("byte functions are available only for byte tensors");
#endif
}

template<>
int THCTensor<real>::logicalany() {
#if defined(TH_REAL_IS_BYTE)
  return THCTensor_(logicalany)(state, tensor);
#else
  throw std::runtime_error("byte functions are available only for byte tensors");
#endif
}

template<>
auto THCTensor<real>::random(const Generator& _generator) -> THCTensor& {
  throw std::runtime_error("THCTensor::random() not implemented");
  return *this;
}

template<>
auto THCTensor<real>::geometric(const Generator& _generator, double p) -> THCTensor& {
  THCTensor_(geometric)(state, tensor, p);
  return *this;
}

template<>
auto THCTensor<real>::bernoulli(const Generator& _generator, double p) -> THCTensor& {
  THCTensor_(bernoulli)(state, tensor, p);
  return *this;
}

template<>
auto THCTensor<real>::bernoulli_FloatTensor(const Generator& _generator,
                                            const Tensor& p) -> THCTensor& {
  THCTensor_(bernoulli_FloatTensor)(state, tensor, const_float_cast(p).tensor);
  return *this;
}

template<>
auto THCTensor<real>::bernoulli_DoubleTensor(const Generator& _generator,
                                             const Tensor& p) -> THCTensor& {
  THCTensor_(bernoulli_DoubleTensor)(state, tensor, const_double_cast(p).tensor);
  return *this;
}

template<>
auto THCTensor<real>::uniform(const Generator& _generator, double a,
                              double b) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(uniform)(state, tensor, a, b);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::normal(const Generator& _generator, double mean,
                             double stdv) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(normal)(state, tensor, mean, stdv);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::exponential(const Generator& _generator,
                                  double lambda) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(exponential)(state, tensor, lambda);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::cauchy(const Generator& _generator, double median,
                             double sigma) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(cauchy)(state, tensor, median, sigma);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::logNormal(const Generator& _generator, double mean,
                                double stdv) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THCTensor_(logNormal)(state, tensor, mean, stdv);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

// Note: the order of *Tensor and *Prob_dist is reversed compared to
// the declarations in TH/generic/THTensorMath.h, so for instance
// the call:
// THRealTensor_multinomial(r, _generator, prob_dist, n_sample, with_replacement)
// is equivalent to `prob_dist->multinomial(r, _generator, n_sample, with_replacement)`.
// It is done this way so that the first argument can be casted onto a float tensor type.
template<>
auto THCTensor<real>::multinomial(const Tensor& r, const Generator& _generator,
                                  int n_sample,
                                  int with_replacement) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  const THCTensor& r_t = const_long_cast(prob_dist).tensor;
  THCTensor_(multinomial)(state, r_t.tensor, tensor, n_sample, with_replacement);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THCTensor<real>::cat(const std::vector<Tensor*>& src, int dimension) -> THCTensor& {
  int num_inputs = src.size();
  std::vector<tensor_type*> inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs[i] = const_tensor_cast(*src[i]).tensor;
  }
  THCTensor_(catArray)(state, tensor, inputs.data(), num_inputs, dimension);
  return *this;
}

template<>
auto THCTensor<real>::gather(const Tensor& src, int dimension,
                             const Tensor& index) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  const THCTensor<long> &index_t = const_long_cast(index);
  THCTensor_(gather)(state, tensor, src_t.tensor, dimension, index_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::scatter(int dimension, const Tensor& index,
                              const Tensor& src) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  const THCTensor<long> &index_t = const_long_cast(index);
  THCTensor_(scatter)(state, tensor, dimension, index_t.tensor, src_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::scatterFill(int dimension, const Tensor& index,
                                  scalar_type value) -> THCTensor& {
  const THCTensor<long> &index_t = const_long_cast(index);
  THCTensor_(scatterFill)(state, tensor, dimension,
      index_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::dot(const Tensor &src) -> scalar_type {
  const THCTensor &src_t = const_tensor_cast(src);
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
auto THCTensor<real>::medianall() -> scalar_type {
  return uncast_scalar(THCTensor_(medianall)(state, tensor));
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
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(neg)(state, tensor, src_t.tensor);
#else
  throw std::runtime_error("neg is only available for `float` and `double` types");
#endif // defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return *this;
}

template<>
auto THCTensor<real>::cinv(const Tensor &src) -> THCTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(cinv)(state, tensor, src_t.tensor);
#else
  throw std::runtime_error("cinv is only available for `float` and `double` types");
#endif // defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return *this;
}

template<>
auto THCTensor<real>::add(const Tensor &src, scalar_type value) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(add)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::sub(const Tensor &src, scalar_type value) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(sub)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::mul(const Tensor &src, scalar_type value) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(mul)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::div(const Tensor &src, scalar_type value) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(div)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::fmod(const Tensor &src, scalar_type value) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(fmod)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::remainder(const Tensor &src,
                                scalar_type value) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(remainder)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::clamp(const Tensor &src, scalar_type min_value,
                            scalar_type max_value) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(clamp)(state, tensor, src_t.tensor, cast_scalar(min_value),
      cast_scalar(max_value));
  return *this;
}

template<>
auto THCTensor<real>::cadd(const Tensor& src1, scalar_type value,
                           const Tensor& src2) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);

  const THCSTensor<real>* src2_sparse;
  if ((src2_sparse = dynamic_cast<const THCSTensor<real>*>(&src2))) {
    THCSTensor_(spcadd)(state, tensor, src1_t.tensor, cast_scalar(value), src2_sparse->tensor);
  } else {
    const THCTensor &src2_t = const_tensor_cast(src2);
    THCTensor_(cadd)(state, tensor, src1_t.tensor, cast_scalar(value), src2_t.tensor);
  }
  return *this;
}

template<>
auto THCTensor<real>::cadd(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  return cadd(src1, static_cast<scalar_type>(1), src2);
}

template<>
auto THCTensor<real>::csub(const Tensor& src1, scalar_type value,
                           const Tensor& src2) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);
  const THCTensor &src2_t = const_tensor_cast(src2);
  THCTensor_(csub)(state, tensor, src1_t.tensor, cast_scalar(value), src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cmul(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);
  const THCTensor &src2_t = const_tensor_cast(src2);
  THCTensor_(cmul)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cpow(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);
  const THCTensor &src2_t = const_tensor_cast(src2);
  THCTensor_(cpow)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cdiv(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);
  const THCTensor &src2_t = const_tensor_cast(src2);
  THCTensor_(cdiv)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cfmod(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);
  const THCTensor &src2_t = const_tensor_cast(src2);
  THCTensor_(cfmod)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cremainder(const Tensor& src1,
                                 const Tensor& src2) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);
  const THCTensor &src2_t = const_tensor_cast(src2);
  THCTensor_(cremainder)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addcmul(const Tensor& src1, scalar_type value,
                              const Tensor& src2,
                              const Tensor& src3) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);
  const THCTensor &src2_t = const_tensor_cast(src2);
  const THCTensor &src3_t = const_tensor_cast(src3);
  THCTensor_(addcmul)(state, tensor, src1_t.tensor, cast_scalar(value),
      src2_t.tensor, src3_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addcdiv(const Tensor& src1, scalar_type value,
                              const Tensor& src2,
                              const Tensor& src3) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);
  const THCTensor &src2_t = const_tensor_cast(src2);
  const THCTensor &src3_t = const_tensor_cast(src3);
  THCTensor_(addcdiv)(state, tensor, src1_t.tensor, cast_scalar(value),
      src2_t.tensor, src3_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addmv(scalar_type beta, const Tensor& src,
                            scalar_type alpha, const Tensor& mat,
                            const Tensor& vec) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  const THCTensor &mat_t = const_tensor_cast(mat);
  const THCTensor &vec_t = const_tensor_cast(vec);
  THCTensor_(addmv)(state, tensor, cast_scalar(beta), src_t.tensor,
      cast_scalar(alpha), mat_t.tensor, vec_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addmm(scalar_type beta, const Tensor& src,
                            scalar_type alpha, const Tensor& mat1,
                            const Tensor& mat2) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  const THCTensor &mat1_t = const_tensor_cast(mat1);
  const THCTensor &mat2_t = const_tensor_cast(mat2);
  THCTensor_(addmm)(state, tensor, cast_scalar(beta), src_t.tensor,
      cast_scalar(alpha), mat1_t.tensor, mat2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addr(scalar_type beta, const Tensor& src,
                           scalar_type alpha, const Tensor& vec1,
                           const Tensor& vec2) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  const THCTensor &vec1_t = const_tensor_cast(vec1);
  const THCTensor &vec2_t = const_tensor_cast(vec2);
  THCTensor_(addr)(state, tensor, cast_scalar(beta), src_t.tensor,
      cast_scalar(alpha), vec1_t.tensor, vec2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::addbmm(scalar_type beta, const Tensor& src,
                             scalar_type alpha, const Tensor& batch1,
                             const Tensor& batch2) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  const THCTensor &batch1_t = const_tensor_cast(batch1);
  const THCTensor &batch2_t = const_tensor_cast(batch2);
  THCTensor_(addbmm)(state, tensor, cast_scalar(beta), src_t.tensor,
      cast_scalar(alpha), batch1_t.tensor, batch2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::baddbmm(scalar_type beta, const Tensor& src,
                              scalar_type alpha, const Tensor& batch1,
                              const Tensor& batch2) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  const THCTensor &batch1_t = const_tensor_cast(batch1);
  const THCTensor &batch2_t = const_tensor_cast(batch2);
  THCTensor_(baddbmm)(state, tensor, cast_scalar(beta), src_t.tensor,
      cast_scalar(alpha), batch1_t.tensor, batch2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::match(const Tensor& m1, const Tensor& m2,
    scalar_type gain) -> THCTensor& {
  throw std::runtime_error("unsupported operation 'match'");
}

template<>
auto THCTensor<real>::max(const Tensor& indices_, const Tensor& src,
                          int dimension, int keepdim) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  const THCTensor<long> &indices__t = const_long_cast(indices_);
  THCTensor_(max)(state, tensor, indices__t.tensor, src_t.tensor, dimension, keepdim);
  return *this;
}

template<>
auto THCTensor<real>::min(const Tensor& indices_, const Tensor& src,
                          int dimension, int keepdim) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  const THCTensor<long> &indices__t = const_long_cast(indices_);
  THCTensor_(min)(state, tensor, indices__t.tensor, src_t.tensor, dimension, keepdim);
  return *this;
}

template<>
auto THCTensor<real>::kthvalue(const Tensor& indices_, const Tensor& src,
                               long k, int dimension, int keepdim) -> THCTensor& {
  throw std::runtime_error("unsupported operation 'kthvalue'");
}

template<>
auto THCTensor<real>::mode(const Tensor& indices_, const Tensor& src,
                           int dimension, int keepdim) -> THCTensor& {
  throw std::runtime_error("unsupported operation 'mode'");
}

template<>
auto THCTensor<real>::median(const Tensor& indices_, const Tensor& src,
                             int dimension, int keepdim) -> THCTensor& {
  throw std::runtime_error("unsupported operation 'median'");
}

template<>
auto THCTensor<real>::sum(const Tensor& src, int dimension, int keepdim) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(sum)(state, tensor, src_t.tensor, dimension, keepdim);
  return *this;
}

template<>
auto THCTensor<real>::prod(const Tensor& src, int dimension, int keepdim) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(prod)(state, tensor, src_t.tensor, dimension, keepdim);
  return *this;
}

template<>
auto THCTensor<real>::cumsum(const Tensor& src, int dimension) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(cumsum)(state, tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THCTensor<real>::cumprod(const Tensor& src, int dimension) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(cumprod)(state, tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THCTensor<real>::sign(const Tensor& src) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(sign)(state, tensor, src_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::trace() -> scalar_type {
  return THCTensor_(trace)(state, tensor);
}

template<>
auto THCTensor<real>::cross(const Tensor& src1, const Tensor& src2,
                            int dimension) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);
  const THCTensor &src2_t = const_tensor_cast(src2);
  THCTensor_(cross)(state, tensor, src1_t.tensor, src2_t.tensor, dimension);
  return *this;
}

template<>
auto THCTensor<real>::cmax(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);
  const THCTensor &src2_t = const_tensor_cast(src2);
  THCTensor_(cmax)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cmin(const Tensor& src1, const Tensor& src2) -> THCTensor& {
  const THCTensor &src1_t = const_tensor_cast(src1);
  const THCTensor &src2_t = const_tensor_cast(src2);
  THCTensor_(cmin)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCTensor<real>::cmaxValue(const Tensor& src,
                                scalar_type value) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
  THCTensor_(cmaxValue)(state, tensor, src_t.tensor, cast_scalar(value));
  return *this;
}

template<>
auto THCTensor<real>::cminValue(const Tensor& src,
                                scalar_type value) -> THCTensor& {
  const THCTensor &src_t = const_tensor_cast(src);
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

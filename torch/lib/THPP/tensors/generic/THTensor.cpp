#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "tensors/generic/THTensor.cpp"
#else

#define const_tensor_cast(tensor) \
  dynamic_cast<const THTensor&>(tensor)
#define const_storage_cast(storage) \
  dynamic_cast<const THStorage<real>&>(storage)
#define const_long_cast(tensor) \
  dynamic_cast<const THTensor<std::conditional<sizeof(long) == 8, long, int64_t>::type>&>(tensor)
#define const_float_cast(tensor) \
  dynamic_cast<const THTensor<float>&>(tensor)
#define const_double_cast(tensor) \
  dynamic_cast<const THTensor<double>&>(tensor)
#define const_int_cast(tensor) \
  dynamic_cast<const THTensor<int32_t>&>(tensor)
#define const_byte_cast(tensor) \
  dynamic_cast<const THTensor<uint8_t>&>(tensor)
#define const_generator_cast(generator) \
  dynamic_cast<const THGenerator&>(generator)

template<>
THTensor<real>::THTensor():
  tensor(THTensor_(new)())
  {};

template<>
THTensor<real>::THTensor(const Tensor& other):
  tensor(THTensor_(newWithTensor)(const_tensor_cast(other).tensor))
  {};

template<>
THTensor<real>::THTensor(const Storage& storage, ptrdiff_t storageOffset,
                         THLongStorage *size, THLongStorage *stride):
  tensor(THTensor_(newWithStorage)(const_storage_cast(storage).storage,
                                   storageOffset, size, stride))
  {};

template<>
THTensor<real>::THTensor(THLongStorage *size, THLongStorage *stride):
  tensor(THTensor_(newWithSize)(size, stride))
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
auto THTensor<real>::newSelect(int dimension, int64_t sliceIndex) const -> THTensor* {
  return new THTensor(THTensor_(newSelect)(tensor, dimension, sliceIndex));
}

template<>
auto THTensor<real>::newNarrow(int dimension, int64_t firstIndex, int64_t size) const -> THTensor* {
  return new THTensor(THTensor_(newNarrow)(tensor, dimension, firstIndex, size));
}

template<>
auto THTensor<real>::newTranspose(int dimension1, int dimension2) const -> THTensor* {
  return new THTensor(THTensor_(newTranspose)(tensor, dimension1, dimension2));
}

template<>
auto THTensor<real>::newUnfold(int dimension, int64_t size, int64_t step) const -> THTensor* {
  return new THTensor(THTensor_(newUnfold)(tensor, dimension, size, step));
}

template<>
auto THTensor<real>::newExpand(const long_range& size) const -> THTensor* {
  THLongStorage *size_storage = THLongStorage_newWithSize(size.size());
  std::memcpy(size_storage->data, size.data(), sizeof(int64_t) * size.size());
  // TODO this might leak on error
  auto expanded = new THTensor(THTensor_(newExpand)(tensor, size_storage));
  THLongStorage_free(size_storage);
  return expanded;
}

template<>
auto THTensor<real>::newView(const long_range& size) const -> THTensor* {
  THLongStorage *size_storage = THLongStorage_newWithSize(size.size());
  std::memcpy(size_storage->data, size.data(), sizeof(int64_t) * size.size());
  // TODO this might leak on error
  auto viewed = new THTensor(THTensor_(newView)(tensor, size_storage));
  THLongStorage_free(size_storage);
  return viewed;
}

template<>
int THTensor<real>::nDim() const {
  return tensor->nDimension;
}

template<>
auto THTensor<real>::sizes() const -> long_range {
  return std::vector<int64_t>(tensor->size, tensor->size + tensor->nDimension);
}

template<>
const int64_t* THTensor<real>::rawSizes() const {
  return tensor->size;
}

template<>
auto THTensor<real>::strides() const -> long_range {
  return long_range(tensor->stride, tensor->stride + tensor->nDimension);
}

template<>
const int64_t* THTensor<real>::rawStrides() const {
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
int64_t THTensor<real>::numel() const {
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
auto THTensor<real>::resize(const std::initializer_list<int64_t> &new_size)
    -> THTensor& {
  return resize(new_size.begin(), new_size.end());
}

template<>
auto THTensor<real>::resize(const std::vector<int64_t> &new_size) -> THTensor& {
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
  THTensor_(resizeAs)(tensor, const_tensor_cast(src).tensor);
  return *this;
}

template<>
template<typename iterator>
auto THTensor<real>::resize(const iterator& begin,
                            const iterator& end) -> THTensor& {
  THLongStorage *sizes = THLongStorage_newWithSize(std::distance(begin, end));
  int64_t *sizes_d = sizes->data;
  for (auto it = begin; it != end; ++it)
    *sizes_d++ = *it;
  // TODO this might leak on error
  THTensor_(resize)(tensor, sizes, nullptr);
  THLongStorage_free(sizes);
  return *this;
}

template<>
auto THTensor<real>::set(const Tensor& src) -> THTensor& {
  THTensor_(set)(tensor, const_tensor_cast(src).tensor);
  return *this;
}

template<>
auto THTensor<real>::setStorage(const Storage& storage,
                                ptrdiff_t storageOffset,
                                const long_range& size,
                                const long_range& stride) -> THTensor& {
  auto raw_storage = dynamic_cast<const THStorage<real>&>(storage).getRaw();
  int nDimension = size.size();
  auto raw_size = const_cast<int64_t*>(size.data());
  auto raw_stride = const_cast<int64_t*>(stride.empty() ? nullptr : stride.data());
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
    const_storage_cast(storage).storage,
    storageOffset,
    size,
    stride
  );
  return *this;
}

template<>
auto THTensor<real>::narrow(const Tensor& src,
                            int dimension,
                            int64_t firstIndex,
                            int64_t size) -> THTensor& {
  THTensor_(narrow)(
    tensor,
    const_tensor_cast(src).tensor,
    dimension,
    firstIndex,
    size
  );
  return *this;
}

template<>
auto THTensor<real>::select(const Tensor& src, int dimension,
                            int64_t sliceIndex) -> THTensor& {
  THTensor_(select)(
    tensor,
    const_tensor_cast(src).tensor,
    dimension,
    sliceIndex
  );
  return *this;
}

template<>
auto THTensor<real>::transpose(const Tensor& src, int dimension1,
                               int dimension2) -> THTensor& {
  auto src_raw = const_tensor_cast(src).tensor;
  if (tensor != src_raw)
    set(src);
  THTensor_(transpose)(tensor, src_raw, dimension1, dimension2);
  return *this;
}

template<>
auto THTensor<real>::unfold(const Tensor& src, int dimension,
                            int64_t size, int64_t step) -> THTensor& {
  auto src_raw = const_tensor_cast(src).tensor;
  if (tensor != src_raw)
    set(src);
  THTensor_(unfold)(tensor, src_raw, dimension, size, step);
  return *this;
}

template<>
auto THTensor<real>::squeeze(const Tensor& src) -> THTensor& {
  auto src_raw = (dynamic_cast<const THTensor<real>&>(src)).tensor;
  THTensor_(squeeze)(tensor, src_raw);
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
auto THTensor<real>::gesv(const Tensor& ra, const Tensor& b, const Tensor& a) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(gesv)(tensor, const_tensor_cast(ra).tensor, const_tensor_cast(b).tensor,
                  const_tensor_cast(a).tensor);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::trtrs(const Tensor& ra, const Tensor& b, const Tensor& a,
                           const char *uplo, const char *trans, const char *diag) -> THTensor& {

#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(trtrs)(tensor, const_tensor_cast(ra).tensor, const_tensor_cast(b).tensor,
                   const_tensor_cast(a).tensor, uplo, trans, diag);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::gels(const Tensor& ra, const Tensor& b, const Tensor& a) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(gels)(tensor, const_tensor_cast(ra).tensor, const_tensor_cast(b).tensor,
                  const_tensor_cast(a).tensor);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::syev(const Tensor& rv, const Tensor& a,
                          const char *jobz, const char *uplo) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(syev)(tensor, const_tensor_cast(rv).tensor, const_tensor_cast(a).tensor, jobz, uplo);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::geev(const Tensor& rv, const Tensor& a, const char *jobvr) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(geev)(tensor, const_tensor_cast(rv).tensor, const_tensor_cast(a).tensor, jobvr);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::gesvd(const Tensor& rs, const Tensor& rv,
                           const Tensor& a, const char *jobu) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(gesvd)(tensor, const_tensor_cast(rs).tensor, const_tensor_cast(rv).tensor,
                   const_tensor_cast(a).tensor, jobu);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::gesvd2(const Tensor& rs, const Tensor& rv, const Tensor& ra,
                            const Tensor& a, const char *jobu) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(gesvd2)(tensor, const_tensor_cast(rs).tensor, const_tensor_cast(rv).tensor,
                    const_tensor_cast(ra).tensor, const_tensor_cast(a).tensor, jobu);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::getri(const Tensor& a) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(getri)(tensor, const_tensor_cast(a).tensor);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::potrf(const Tensor& a, const char *uplo) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(potrf)(tensor, const_tensor_cast(a).tensor, uplo);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::potrs(const Tensor& b, const Tensor& a, const char *uplo) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(potrs)(tensor, const_tensor_cast(b).tensor, const_tensor_cast(a).tensor, uplo);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::potri(const Tensor& a, const char *uplo) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(potri)(tensor, const_tensor_cast(a).tensor, uplo);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::qr(const Tensor& rr, const Tensor& a) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(qr)(tensor, const_tensor_cast(rr).tensor, const_tensor_cast(a).tensor);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::geqrf(const Tensor& rtau, const Tensor& a) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(geqrf)(tensor, const_tensor_cast(rtau).tensor, const_tensor_cast(a).tensor);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::orgqr(const Tensor& a, const Tensor& tau) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(orgqr)(tensor, const_tensor_cast(a).tensor, const_tensor_cast(tau).tensor);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::ormqr(const Tensor& a, const Tensor& tau, const Tensor& c,
                           const char *side, const char *trans) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(ormqr)(tensor, const_tensor_cast(a).tensor, const_tensor_cast(tau).tensor,
                   const_tensor_cast(c).tensor, side, trans);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::pstrf(const Tensor& rpiv, const Tensor& a,
                           const char *uplo, scalar_type tol) -> THTensor& {
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(pstrf)(tensor, const_int_cast(rpiv).tensor, const_tensor_cast(a).tensor, uplo, tol);
  return *this;
#else
  throw std::runtime_error("Lapack operations are available only for floating point tensors");
#endif
}

template<>
auto THTensor<real>::fill(scalar_type value) -> THTensor& {
  THTensor_(fill)(tensor, value);
  return *this;
}

template<>
auto THTensor<real>::maskedFill(const Tensor& mask, scalar_type value) -> THTensor& {
  THTensor_(maskedFill)(tensor, const_byte_cast(mask).tensor, value);
  return *this;
}

template<>
auto THTensor<real>::maskedCopy(const Tensor& mask, const Tensor& src) -> THTensor& {
  THTensor_(maskedCopy)(tensor, const_byte_cast(mask).tensor, const_tensor_cast(src).tensor);
  return *this;
}

template<>
auto THTensor<real>::maskedSelect(const Tensor& src, const Tensor& mask) -> THTensor& {
  THTensor_(maskedSelect)(tensor, const_tensor_cast(src).tensor, const_byte_cast(mask).tensor);
  return *this;
}

template<>
auto THTensor<real>::nonzero(const Tensor& subscript) -> THTensor& {
  THTensor_(nonzero)(const_long_cast(subscript).tensor, tensor);
  return *this;
}

template<>
auto THTensor<real>::indexSelect(const Tensor& src, int dim, const Tensor& index) -> THTensor& {
  THTensor_(indexSelect)(tensor, const_tensor_cast(src).tensor, dim,
                         const_long_cast(index).tensor);
  return *this;
}

template<>
auto THTensor<real>::indexCopy(int dim, const Tensor& index, const Tensor& src) -> THTensor& {
  THTensor_(indexCopy)(tensor, dim, const_long_cast(index).tensor,
                       const_tensor_cast(src).tensor);
  return *this;
}

template<>
auto THTensor<real>::indexAdd(int dim, const Tensor& index, const Tensor& src) -> THTensor& {
  THTensor_(indexAdd)(tensor, dim, const_long_cast(index).tensor,
                      const_tensor_cast(src).tensor);
  return *this;
}

template<>
auto THTensor<real>::indexFill(int dim, const Tensor& index, scalar_type value) -> THTensor& {
  THTensor_(indexFill)(tensor, dim, const_long_cast(index).tensor, value);
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

template<>
auto THTensor<real>::diag(const Tensor& src, int k) -> THTensor& {
  THTensor_(diag)(tensor, const_tensor_cast(src).tensor, k);
  return *this;
}

template<>
auto THTensor<real>::eye(int64_t n, int64_t m) -> THTensor& {
  THTensor_(eye)(tensor, n, m);
  return *this;
}

template<>
auto THTensor<real>::range(scalar_type xmin, scalar_type xmax,
                           scalar_type step) -> THTensor& {
  THTensor_(range)(tensor, xmin, xmax, step);
  return *this;
}

template<>
auto THTensor<real>::randperm(const Generator& _generator, int64_t n) -> THTensor& {
  THTensor_(randperm)(tensor, const_generator_cast(_generator).generator, n);
  return *this;
}

template<>
auto THTensor<real>::sort(const Tensor& ri, const Tensor& src,
                          int dimension, int desc) -> THTensor& {
  THTensor_(sort)(
    tensor,
    const_long_cast(ri).tensor,
    const_tensor_cast(src).tensor,
    dimension,
    desc
  );
  return *this;
}

template<>
auto THTensor<real>::topk(const Tensor& ri, const Tensor& src,
                          int64_t k, int dim, int dir, int sorted) -> THTensor& {
  THTensor_(topk)(
    tensor,
    const_long_cast(ri).tensor,
    const_tensor_cast(src).tensor,
    k,
    dim,
    dir,
    sorted
  );
  return *this;
}

template<>
auto THTensor<real>::tril(const Tensor& src, int64_t k) -> THTensor& {
  THTensor_(tril)(tensor, const_tensor_cast(src).tensor, k);
  return *this;
}

template<>
auto THTensor<real>::triu(const Tensor& src, int64_t k) -> THTensor& {
  THTensor_(triu)(tensor, const_tensor_cast(src).tensor, k);
  return *this;
}

template<>
auto THTensor<real>::catArray(const std::vector<Tensor*>& inputs_vec,
                              int dimension) -> THTensor& {
  int numInputs = inputs_vec.size();
  // TOFIX: workaround for variable length arrays in MSVC
  tensor_type **inputs = (tensor_type **) alloca(numInputs * sizeof(tensor_type*));
  for (std::size_t i = 0; i < numInputs; i++)
    inputs[i] = const_tensor_cast(*inputs_vec[i]).tensor;
  THTensor_(catArray)(tensor, inputs, numInputs, dimension);
  return *this;
}

template<>
int THTensor<real>::equal(const Tensor& other) const {
  return THTensor_(equal)(tensor, const_tensor_cast(other).tensor);
}

  // Note: the order in *Value and *Tensor is reversed compared to
  // the declarations in TH/generic/THTensorMath.h, so for instance
  // the call THRealTensor_ltTensor(r, ta, tb) is equivalent to
  // ta->ltTensor(r, tb). It is done this way so that the first
  // argument can be casted onto a byte tensor type

#define TENSOR_IMPLEMENT_LOGICAL(NAME)                               \
  template<>                                                         \
  auto THTensor<real>::NAME##Value(const Tensor& r,                  \
                                   scalar_type value) -> THTensor& { \
    if (r.type() != Type::UCHAR)                                     \
      throw std::invalid_argument("logical operator called on non-byte tensor"); \
    THTensor_(NAME##Value)(                                          \
      const_byte_cast(r).tensor,                                           \
      tensor,                                                        \
      value                                                          \
    );                                                               \
    return *this;                                                    \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THTensor<real>::NAME##ValueT(const Tensor& t,                 \
                                   scalar_type value) -> THTensor& { \
    THTensor_(NAME##ValueT)(                                         \
      tensor,                                                        \
      const_tensor_cast(t).tensor,                                         \
      value                                                          \
    );                                                               \
    return *this;                                                    \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THTensor<real>::NAME##Tensor(const Tensor& r,                 \
                                    const Tensor& tb) -> THTensor& { \
    if (r.type() != Type::UCHAR)                                     \
      throw std::invalid_argument("logical operator called on non-byte tensor"); \
    THTensor_(NAME##Tensor)(                                         \
      const_byte_cast(r).tensor,                                     \
      tensor,                                                        \
      const_tensor_cast(tb).tensor                                   \
    );                                                               \
    return *this;                                                    \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THTensor<real>::NAME##TensorT(const Tensor& ta,               \
                                     const Tensor& tb) -> THTensor& { \
    THTensor_(NAME##TensorT)(                                        \
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
auto THTensor<real>::abs(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_INT) ||\
    defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  THTensor_(abs)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("absolute value is only available for\
      signed type tensors");
#endif
}

template<>
auto THTensor<real>::sigmoid(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(sigmoid)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::log(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(log)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::log1p(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(log1p)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::exp(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(exp)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::cos(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(cos)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::acos(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(acos)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::cosh(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(cosh)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::sin(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(sin)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::asin(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(asin)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::sinh(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(sinh)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::copy(const Tensor& src) -> THTensor& {
  // TODO: polymorphic copy
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(copy)(tensor, src_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::tan(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(tan)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::atan(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(atan)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::atan2(const Tensor& src1, const Tensor& src2) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(atan2)(
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
auto THTensor<real>::tanh(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(tanh)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::pow(const Tensor& src, scalar_type value) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(pow)(tensor, const_tensor_cast(src).tensor, value);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::tpow(scalar_type value, const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(tpow)(tensor, value, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::sqrt(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(sqrt)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::rsqrt(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(rsqrt)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::ceil(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(ceil)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::floor(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(floor)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::round(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(round)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::trunc(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(trunc)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::frac(const Tensor& src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(frac)(tensor, const_tensor_cast(src).tensor);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::lerp(const Tensor& a, const Tensor& b, scalar_type weight) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(lerp)(
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
auto THTensor<real>::mean(const Tensor& src, int dimension, int keepdim) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(mean)(tensor, const_tensor_cast(src).tensor, dimension, keepdim);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::std(const Tensor& src, int dimension, int biased, int keepdim) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(std)(tensor, const_tensor_cast(src).tensor, dimension, biased, keepdim);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::var(const Tensor& src, int dimension, int biased, int keepdim) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(var)(tensor, const_tensor_cast(src).tensor, dimension, biased, keepdim);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::norm(const Tensor& src,
                          scalar_type value,
                          int dimension,
                          int keepdim) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(norm)(tensor, const_tensor_cast(src).tensor, value, dimension, keepdim);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::renorm(const Tensor& src,
                            scalar_type value,
                            int dimension,
                            scalar_type maxnorm) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(renorm)(tensor, const_tensor_cast(src).tensor, value, dimension, maxnorm);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::histc(const Tensor& src,
                           int64_t nbins,
                           scalar_type minvalue,
                           scalar_type maxvalue) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  const THTensor& src_t = const_tensor_cast(src);
  THTensor_(histc)(tensor, src_t.tensor, nbins, minvalue, maxvalue);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::bhistc(const Tensor& src,
                            int64_t nbins,
                            scalar_type minvalue,
                            scalar_type maxvalue) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  const THTensor& src_t = const_tensor_cast(src);
  THTensor_(bhistc)(tensor, src_t.tensor, nbins, minvalue, maxvalue);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::dist(const Tensor& src, scalar_type value) -> scalar_type {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return THTensor_(dist)(tensor, const_tensor_cast(src).tensor, value);
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::meanall() -> scalar_type {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return THTensor_(meanall)(tensor);
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::varall(int biased) -> scalar_type{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return THTensor_(varall)(tensor, biased);
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::stdall(int biased) -> scalar_type {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return THTensor_(stdall)(tensor, biased);
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::normall(scalar_type value) -> scalar_type {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return THTensor_(normall)(tensor, value);
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::linspace(scalar_type a, scalar_type b, int64_t n) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(linspace)(tensor, a, b, n);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::logspace(scalar_type a, scalar_type b, int64_t n) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(logspace)(tensor, a, b, n);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::rand(const Generator& _generator, THLongStorage *size) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(rand)(tensor, const_generator_cast(_generator).generator, size);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::randn(const Generator& _generator, THLongStorage *size) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(randn)(tensor, const_generator_cast(_generator).generator, size);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
int THTensor<real>::logicalall() {
#if defined(TH_REAL_IS_BYTE)
  return THTensor_(logicalall)(tensor);
#else
  throw std::runtime_error("byte functions are available only for byte tensors");
#endif
}

template<>
int THTensor<real>::logicalany() {
#if defined(TH_REAL_IS_BYTE)
  return THTensor_(logicalany)(tensor);
#else
  throw std::runtime_error("byte functions are available only for byte tensors");
#endif
}

template<>
auto THTensor<real>::random(const Generator& _generator) -> THTensor& {
  THTensor_(random)(tensor, const_generator_cast(_generator).generator);
  return *this;
}

template<>
auto THTensor<real>::geometric(const Generator& _generator, double p) -> THTensor& {
  THTensor_(geometric)(tensor, const_generator_cast(_generator).generator, p);
  return *this;
}

template<>
auto THTensor<real>::bernoulli(const Generator& _generator, double p) -> THTensor& {
  THTensor_(bernoulli)(tensor, const_generator_cast(_generator).generator, p);
  return *this;
}

template<>
auto THTensor<real>::bernoulli_FloatTensor(const Generator& _generator,
                                           const Tensor& p) -> THTensor& {
  THTensor_(bernoulli_FloatTensor)(
    tensor,
    const_generator_cast(_generator).generator,
    const_float_cast(p).tensor
  );
  return *this;
}

template<>
auto THTensor<real>::bernoulli_DoubleTensor(const Generator& _generator,
                                            const Tensor& p) -> THTensor& {
  THTensor_(bernoulli_DoubleTensor)(
    tensor,
    const_generator_cast(_generator).generator,
    const_double_cast(p).tensor
  );
  return *this;
}

template<>
auto THTensor<real>::uniform(const Generator& _generator, double a, double b) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(uniform)(tensor, const_generator_cast(_generator).generator, a, b);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::normal(const Generator& _generator, double mean, double stdv) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(normal)(tensor, const_generator_cast(_generator).generator, mean, stdv);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::exponential(const Generator& _generator, double lambda) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(exponential)(tensor, const_generator_cast(_generator).generator, lambda);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::cauchy(const Generator& _generator,
                            double median,
                            double sigma) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(cauchy)(tensor, const_generator_cast(_generator).generator, median, sigma);
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::logNormal(const Generator& _generator,
                               double mean,
                               double stdv) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(logNormal)(tensor, const_generator_cast(_generator).generator, mean, stdv);
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
auto THTensor<real>::multinomial(const Tensor& r, const Generator& _generator,
                                 int n_sample, int with_replacement) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  THTensor_(multinomial)(
    const_long_cast(r).tensor,
    const_generator_cast(_generator).generator,
    tensor,
    n_sample,
    with_replacement
  );
  return *this;
#else
  throw std::runtime_error("floating point functions are available only for\
      floating point tensors");
#endif
}

template<>
auto THTensor<real>::cat(const std::vector<Tensor*>& src, int dimension) -> THTensor& {
  int num_inputs = src.size();
  std::vector<tensor_type*> inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs[i] = const_tensor_cast(*src[i]).tensor;
  }
  THTensor_(catArray)(tensor, inputs.data(), num_inputs, dimension);
  return *this;
}

template<>
auto THTensor<real>::gather(const Tensor& src, int dimension,
                            const Tensor& index) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<int64_t> &index_t = const_long_cast(index);
  THTensor_(gather)(tensor, src_t.tensor, dimension, index_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::scatter(int dimension, const Tensor& index,
                             const Tensor& src) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<int64_t> &index_t = const_long_cast(index);
  THTensor_(scatter)(tensor, dimension, index_t.tensor, src_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::scatterFill(int dimension, const Tensor& index,
                                 scalar_type value) -> THTensor& {
  const THTensor<int64_t> &index_t = const_long_cast(index);
  THTensor_(scatterFill)(tensor, dimension, index_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::dot(const Tensor &src) -> scalar_type {
  const THTensor &src_t = const_tensor_cast(src);
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
auto THTensor<real>::medianall() -> scalar_type {
  return THTensor_(medianall)(tensor);
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
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(neg)(tensor, src_t.tensor);
#else
  throw std::runtime_error("neg is only available for `float` and `double` types");
#endif // defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return *this;
}

template<>
auto THTensor<real>::cinv(const Tensor &src) -> THTensor& {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(cinv)(tensor, src_t.tensor);
#else
  throw std::runtime_error("cinv is only available for `float` and `double` types");
#endif // defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return *this;
}

template<>
auto THTensor<real>::add(const Tensor &src, scalar_type value) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(add)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::sub(const Tensor &src, scalar_type value) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(sub)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::mul(const Tensor &src, scalar_type value) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(mul)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::div(const Tensor &src, scalar_type value) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(div)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::fmod(const Tensor &src, scalar_type value) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(fmod)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::remainder(const Tensor &src, scalar_type value) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(remainder)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::clamp(const Tensor &src, scalar_type min_value,
                           scalar_type max_value) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(clamp)(tensor, src_t.tensor, min_value, max_value);
  return *this;
}

template<>
auto THTensor<real>::cadd(const Tensor& src1, scalar_type value, const Tensor& src2) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);

  const THSTensor<real>* src2_sparse;
  if ((src2_sparse = dynamic_cast<const THSTensor<real>*>(&src2))) {
    THSTensor_(spcadd)(tensor, src1_t.tensor, value, src2_sparse->tensor);
  } else {
    const THTensor &src2_t = const_tensor_cast(src2);
    THTensor_(cadd)(tensor, src1_t.tensor, value, src2_t.tensor);
  }
  return *this;
}

template<>
auto THTensor<real>::cadd(const Tensor& src1, const Tensor& src2) -> THTensor& {
  return cadd(src1, static_cast<scalar_type>(1), src2);
}

template<>
auto THTensor<real>::csub(const Tensor& src1, scalar_type value,
                          const Tensor& src2) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);
  const THTensor &src2_t = const_tensor_cast(src2);
  THTensor_(csub)(tensor, src1_t.tensor, value, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cmul(const Tensor& src1, const Tensor& src2) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);
  const THTensor &src2_t = const_tensor_cast(src2);
  THTensor_(cmul)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cpow(const Tensor& src1, const Tensor& src2) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);
  const THTensor &src2_t = const_tensor_cast(src2);
  THTensor_(cpow)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cdiv(const Tensor& src1, const Tensor& src2) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);
  const THTensor &src2_t = const_tensor_cast(src2);
  THTensor_(cdiv)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cfmod(const Tensor& src1, const Tensor& src2) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);
  const THTensor &src2_t = const_tensor_cast(src2);
  THTensor_(cfmod)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cremainder(const Tensor& src1,
                                const Tensor& src2) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);
  const THTensor &src2_t = const_tensor_cast(src2);
  THTensor_(cremainder)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addcmul(const Tensor& src1, scalar_type value,
                             const Tensor& src2, const Tensor& src3) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);
  const THTensor &src2_t = const_tensor_cast(src2);
  const THTensor &src3_t = const_tensor_cast(src3);
  THTensor_(addcmul)(tensor, src1_t.tensor, value, src2_t.tensor, src3_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addcdiv(const Tensor& src1, scalar_type value,
                             const Tensor& src2, const Tensor& src3) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);
  const THTensor &src2_t = const_tensor_cast(src2);
  const THTensor &src3_t = const_tensor_cast(src3);
  THTensor_(addcdiv)(tensor, src1_t.tensor, value, src2_t.tensor, src3_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addmv(scalar_type beta, const Tensor& src,
                           scalar_type alpha, const Tensor& mat,
                           const Tensor& vec) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor &mat_t = const_tensor_cast(mat);
  const THTensor &vec_t = const_tensor_cast(vec);
  THTensor_(addmv)(tensor, beta, src_t.tensor, alpha, mat_t.tensor, vec_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addmm(scalar_type beta, const Tensor& src,
                           scalar_type alpha, const Tensor& mat1,
                           const Tensor& mat2) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor &mat1_t = const_tensor_cast(mat1);
  const THTensor &mat2_t = const_tensor_cast(mat2);
  THTensor_(addmm)(tensor, beta, src_t.tensor, alpha, mat1_t.tensor, mat2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addr(scalar_type beta, const Tensor& src,
                          scalar_type alpha, const Tensor& vec1,
                          const Tensor& vec2) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor &vec1_t = const_tensor_cast(vec1);
  const THTensor &vec2_t = const_tensor_cast(vec2);
  THTensor_(addr)(tensor, beta, src_t.tensor, alpha, vec1_t.tensor, vec2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::addbmm(scalar_type beta, const Tensor& src,
                            scalar_type alpha, const Tensor& batch1,
                            const Tensor& batch2) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor &batch1_t = const_tensor_cast(batch1);
  const THTensor &batch2_t = const_tensor_cast(batch2);
  THTensor_(addbmm)(tensor, beta, src_t.tensor, alpha, batch1_t.tensor,
      batch2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::baddbmm(scalar_type beta, const Tensor& src,
                             scalar_type alpha, const Tensor& batch1,
                             const Tensor& batch2) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor &batch1_t = const_tensor_cast(batch1);
  const THTensor &batch2_t = const_tensor_cast(batch2);
  THTensor_(baddbmm)(tensor, beta, src_t.tensor, alpha, batch1_t.tensor,
      batch2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::match(const Tensor& m1, const Tensor& m2,
                           scalar_type gain) -> THTensor& {
  const THTensor &m1_t = const_tensor_cast(m1);
  const THTensor &m2_t = const_tensor_cast(m2);
  THTensor_(match)(tensor, m1_t.tensor, m2_t.tensor, gain);
  return *this;
}

template<>
auto THTensor<real>::max(const Tensor& indices_, const Tensor& src,
                         int dimension, int keepdim) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<int64_t> &indices__t = const_long_cast(indices_);
  THTensor_(max)(tensor, indices__t.tensor, src_t.tensor, dimension, keepdim);
  return *this;
}

template<>
auto THTensor<real>::min(const Tensor& indices_, const Tensor& src,
                         int dimension, int keepdim) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<int64_t> &indices__t = const_long_cast(indices_);
  THTensor_(min)(tensor, indices__t.tensor, src_t.tensor, dimension, keepdim);
  return *this;
}

template<>
auto THTensor<real>::kthvalue(const Tensor& indices_, const Tensor& src,
                              int64_t k, int dimension, int keepdim) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<int64_t> &indices__t = const_long_cast(indices_);
  THTensor_(kthvalue)(tensor, indices__t.tensor, src_t.tensor, k, dimension, keepdim);
  return *this;
}

template<>
auto THTensor<real>::mode(const Tensor& indices_, const Tensor& src,
                          int dimension, int keepdim) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<int64_t> &indices__t = const_long_cast(indices_);
  THTensor_(mode)(tensor, indices__t.tensor, src_t.tensor, dimension, keepdim);
  return *this;
}

template<>
auto THTensor<real>::median(const Tensor& indices_, const Tensor& src,
                            int dimension, int keepdim) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<int64_t> &indices__t = const_long_cast(indices_);
  THTensor_(median)(tensor, indices__t.tensor, src_t.tensor, dimension, keepdim);
  return *this;
}

template<>
auto THTensor<real>::sum(const Tensor& src, int dimension, int keepdim) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(sum)(tensor, src_t.tensor, dimension, keepdim);
  return *this;
}

template<>
auto THTensor<real>::prod(const Tensor& src, int dimension, int keepdim) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(prod)(tensor, src_t.tensor, dimension, keepdim);
  return *this;
}

template<>
auto THTensor<real>::cumsum(const Tensor& src, int dimension) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(cumsum)(tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::cumprod(const Tensor& src, int dimension) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(cumprod)(tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::sign(const Tensor& src) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(sign)(tensor, src_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::trace() -> scalar_type {
  return THTensor_(trace)(tensor);
}

template<>
auto THTensor<real>::cross(const Tensor& src1, const Tensor& src2,
                           int dimension) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);
  const THTensor &src2_t = const_tensor_cast(src2);
  THTensor_(cross)(tensor, src1_t.tensor, src2_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::cmax(const Tensor& src1, const Tensor& src2) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);
  const THTensor &src2_t = const_tensor_cast(src2);
  THTensor_(cmax)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cmin(const Tensor& src1, const Tensor& src2) -> THTensor& {
  const THTensor &src1_t = const_tensor_cast(src1);
  const THTensor &src2_t = const_tensor_cast(src2);
  THTensor_(cmin)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::cmaxValue(const Tensor& src,
                               scalar_type value) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(cmaxValue)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THTensor<real>::cminValue(const Tensor& src,
                               scalar_type value) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
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

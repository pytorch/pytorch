#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "tensors/generic/THTensor.cpp"
#else

#define const_tensor_cast(tensor) \
  dynamic_cast<const THTensor&>(tensor)
#define const_storage_cast(storage) \
  dynamic_cast<const THStorage<real>&>(storage)
#define const_long_cast(tensor) \
  dynamic_cast<const THTensor<long>&>(tensor)
#define const_byte_cast(tensor) \
  dynamic_cast<const THTensor<unsigned char>&>(tensor)

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
auto THTensor<real>::resize(const std::initializer_list<long> &new_size)
    -> THTensor& {
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
  THTensor_(resizeAs)(tensor, const_tensor_cast(src).tensor);
  return *this;
}

template<>
template<typename iterator>
auto THTensor<real>::resize(const iterator& begin,
                            const iterator& end) -> THTensor& {
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
                            long firstIndex,
                            long size) -> THTensor& {
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
                            long sliceIndex) -> THTensor& {
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
                            long size, long step) -> THTensor& {
  auto src_raw = const_tensor_cast(src).tensor;
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

template<>
auto THTensor<real>::diag(const Tensor& src, int k) -> THTensor& {
  THTensor_(diag)(tensor, const_tensor_cast(src).tensor, k);
  return *this;
}

template<>
auto THTensor<real>::eye(long n, long m) -> THTensor& {
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
                          long k, int dim, int dir, int sorted) -> THTensor& {
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
auto THTensor<real>::tril(const Tensor& src, long k) -> THTensor& {
  THTensor_(tril)(tensor, const_tensor_cast(src).tensor, k);
  return *this;
}

template<>
auto THTensor<real>::triu(const Tensor& src, long k) -> THTensor& {
  THTensor_(triu)(tensor, const_tensor_cast(src).tensor, k);
  return *this;
}

template<>
auto THTensor<real>::catArray(const std::vector<Tensor*>& inputs_vec,
                              int dimension) -> THTensor& {
  int numInputs = inputs_vec.size();
  tensor_type *inputs[numInputs];
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
  const THTensor<long> &index_t = const_long_cast(index);
  THTensor_(gather)(tensor, src_t.tensor, dimension, index_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::scatter(int dimension, const Tensor& index,
                             const Tensor& src) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<long> &index_t = const_long_cast(index);
  THTensor_(scatter)(tensor, dimension, index_t.tensor, src_t.tensor);
  return *this;
}

template<>
auto THTensor<real>::scatterFill(int dimension, const Tensor& index,
                                 scalar_type value) -> THTensor& {
  const THTensor<long> &index_t = const_long_cast(index);
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
    return *this;
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
                         int dimension) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<long> &indices__t = const_long_cast(indices_);
  THTensor_(max)(tensor, indices__t.tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::min(const Tensor& indices_, const Tensor& src,
                         int dimension) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<long> &indices__t = const_long_cast(indices_);
  THTensor_(min)(tensor, indices__t.tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::kthvalue(const Tensor& indices_, const Tensor& src,
                              long k, int dimension) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<long> &indices__t = const_long_cast(indices_);
  THTensor_(kthvalue)(tensor, indices__t.tensor, src_t.tensor, k, dimension);
  return *this;
}

template<>
auto THTensor<real>::mode(const Tensor& indices_, const Tensor& src,
                          int dimension) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<long> &indices__t = const_long_cast(indices_);
  THTensor_(mode)(tensor, indices__t.tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::median(const Tensor& indices_, const Tensor& src,
                            int dimension) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  const THTensor<long> &indices__t = const_long_cast(indices_);
  THTensor_(median)(tensor, indices__t.tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::sum(const Tensor& src, int dimension) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(sum)(tensor, src_t.tensor, dimension);
  return *this;
}

template<>
auto THTensor<real>::prod(const Tensor& src, int dimension) -> THTensor& {
  const THTensor &src_t = const_tensor_cast(src);
  THTensor_(prod)(tensor, src_t.tensor, dimension);
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

#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "tensors/generic/THSTensor.cpp"
#else

#define const_tensor_cast(tensor) \
  dynamic_cast<const THSTensor&>(tensor)
#define const_storage_cast(storage) \
  dynamic_cast<const THStorage<real>&>(storage)
#define const_long_cast(tensor) \
  dynamic_cast<const THSTensor<long>&>(tensor)
#define const_byte_cast(tensor) \
  dynamic_cast<const THSTensor<unsigned char>&>(tensor)

template<>
THSTensor<real>::THSTensor():
  tensor(THSTensor_(new)())
  {};

template<>
THSTensor<real>::THSTensor(THSRealTensor *wrapped):
  tensor(wrapped)
  {};

template<>
THSTensor<real>::~THSTensor() {
  if (tensor)
    THSTensor_(free)(tensor);
}

template<>
auto THSTensor<real>::clone() const -> THSTensor* {
  return new THSTensor(THSTensor_(newClone)(tensor));
}

template<>
auto THSTensor<real>::clone_shallow() -> THSTensor* {
  THSTensor_(retain)(tensor);
  return new THSTensor(tensor);
}

template<>
auto THSTensor<real>::contiguous() const -> std::unique_ptr<Tensor> {
  return std::unique_ptr<Tensor>(new THSTensor(THSTensor_(newContiguous)(tensor)));
}

template<>
int THSTensor<real>::nDim() const {
  return tensor->nDimensionI;
}

template<>
auto THSTensor<real>::sizes() const -> long_range {
  return std::vector<long>(tensor->size, tensor->size + tensor->nDimensionI);
}

template<>
const long* THSTensor<real>::rawSizes() const {
  return tensor->size;
}

template<>
auto THSTensor<real>::strides() const -> long_range {
  throw std::runtime_error("THSTensor::strides() not supported");
}

template<>
const long* THSTensor<real>::rawStrides() const {
  throw std::runtime_error("THSTensor::rawStrides() not supported");
}

template<>
std::size_t THSTensor<real>::storageOffset() const {
  throw std::runtime_error("THSTensor::storageOffset() not supported");
}

template<>
std::size_t THSTensor<real>::elementSize() const {
  return sizeof(real);
}

template<>
long long THSTensor<real>::numel() const {
  throw std::runtime_error("THSTensor::numel not supported");
}

template<>
bool THSTensor<real>::isContiguous() const {
  throw std::runtime_error("THSTensor::isContiguous() not supported");
}

template<>
void* THSTensor<real>::data() {
  throw std::runtime_error("THSTensor::data() not supported");
}

template<>
const void* THSTensor<real>::data() const {
  throw std::runtime_error("THSTensor::data() not supported");
}

template<>
void* THSTensor<real>::cdata() {
  return tensor;
}

template<>
const void* THSTensor<real>::cdata() const {
  return tensor;
}

template<>
auto THSTensor<real>::resize(const std::initializer_list<long> &new_size) -> THSTensor& {
  throw std::runtime_error("THSTensor::resize() not supported");
}

template<>
auto THSTensor<real>::resize(const std::vector<long> &new_size) -> THSTensor& {
  throw std::runtime_error("THSTensor::resize() not supported");
}

template<>
auto THSTensor<real>::resize(THLongStorage *size, THLongStorage *stride) -> THSTensor& {
  throw std::runtime_error("THSTensor::resize() not supported");
}

template<>
auto THSTensor<real>::resizeAs(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::resizeAs() not supported");
}

template<>
template<typename iterator>
auto THSTensor<real>::resize(const iterator& begin, const iterator& end) -> THSTensor& {
  throw std::runtime_error("THSTensor::resize() not supported");
}

template<>
auto THSTensor<real>::set(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::set() not supported");
}

template<>
auto THSTensor<real>::setStorage(const Storage& storage,
                                 ptrdiff_t storageOffset,
                                 const long_range& size,
                                 const long_range& stride) -> THSTensor& {
  throw std::runtime_error("THSTensor::setStorage not supported");
}

template<>
auto THSTensor<real>::setStorage(const Storage& storage,
                                 ptrdiff_t storageOffset,
                                 THLongStorage *size,
                                 THLongStorage *stride) -> THSTensor& {
  throw std::runtime_error("THSTensor::setStorage not supported");
}

template<>
auto THSTensor<real>::narrow(const Tensor& src,
                             int dimension,
                             long firstIndex,
                             long size) -> THSTensor& {
  throw std::runtime_error("THSTensor::narrow not supported");
}

template<>
auto THSTensor<real>::select(const Tensor& src, int dimension,
                             long sliceIndex) -> THSTensor& {
  throw std::runtime_error("THSTensor::select not supported");
}

template<>
auto THSTensor<real>::transpose(const Tensor& src, int dimension1,
                                int dimension2) -> THSTensor& {
  throw std::runtime_error("THSTensor::transpose not supported");
}


template<>
auto THSTensor<real>::unfold(const Tensor& src, int dimension,
                             long size, long step) ->THSTensor& {
  throw std::runtime_error("THSTensor::unfold not supported");
}

template<>
auto THSTensor<real>::squeeze(const Tensor& src, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::squeeze not supported");
}

template<>
auto THSTensor<real>::unsqueeze(const Tensor& src, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::unsqueeze not supported");
}

template<>
auto THSTensor<real>::fill(scalar_type value) -> THSTensor& {
  throw std::runtime_error("THSTensor::fill() not supported");
}

template<>
auto THSTensor<real>::retain() -> THSTensor& {
  THSTensor_(retain)(tensor);
  return *this;
}

template<>
auto THSTensor<real>::free() -> THSTensor& {
  THSTensor_(free)(tensor);
  return *this;
}

template<>
auto THSTensor<real>::diag(const Tensor& src, int k) -> THSTensor& {
  throw std::runtime_error("THSTensor::diag() not supported");
}

template<>
auto THSTensor<real>::eye(long n, long m) -> THSTensor& {
  throw std::runtime_error("THSTensor::eye() not supported");
}

template<>
auto THSTensor<real>::range(scalar_type xmin, scalar_type xmax,
                           scalar_type step) -> THSTensor& {
  throw std::runtime_error("THSTensor::range() not supported");
}

template<>
auto THSTensor<real>::sort(const Tensor& ri, const Tensor& src,
                          int dimension, int desc) -> THSTensor& {
  throw std::runtime_error("THSTensor::sort() not supported");
}

template<>
auto THSTensor<real>::topk(const Tensor& ri, const Tensor& src,
                          long k, int dim, int dir, int sorted) -> THSTensor& {
  throw std::runtime_error("THSTensor::topk() not supported");
}

template<>
auto THSTensor<real>::tril(const Tensor& src, long k) -> THSTensor& {
  throw std::runtime_error("THSTensor::tril() not supported");
}

template<>
auto THSTensor<real>::triu(const Tensor& src, long k) -> THSTensor& {
  throw std::runtime_error("THSTensor::triu() not supported");
}

template<>
auto THSTensor<real>::catArray(const std::vector<Tensor*>& inputs_vec,
                              int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::catArray() not supported");
}

template<>
int THSTensor<real>::equal(const Tensor& other) const {
  throw std::runtime_error("THSTensor::equal() not supported");
}

  // Note: the order in *Value and *Tensor is reversed compared to
  // the declarations in TH/generic/THSTensorMath.h, so for instance
  // the call THRealTensor_ltTensor(r, ta, tb) is equivalent to
  // ta->ltTensor(r, tb). It is done this way so that the first
  // argument can be casted onto a byte tensor type

#define TENSOR_IMPLEMENT_LOGICAL(NAME)                               \
  template<>                                                         \
  auto THSTensor<real>::NAME##Value(const Tensor& r,                  \
                                   scalar_type value) -> THSTensor& { \
    throw std::invalid_argument("THSTensor::" #NAME "Value() not supported"); \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THSTensor<real>::NAME##ValueT(const Tensor& t,                 \
                                   scalar_type value) -> THSTensor& { \
    throw std::invalid_argument("THSTensor::" #NAME "ValueT() not supported"); \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THSTensor<real>::NAME##Tensor(const Tensor& r,                 \
                                    const Tensor& tb) -> THSTensor& { \
    throw std::invalid_argument("THSTensor::" #NAME "Tensor() not supported"); \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THSTensor<real>::NAME##TensorT(const Tensor& ta,               \
                                     const Tensor& tb) -> THSTensor& { \
    throw std::invalid_argument("THSTensor::" #NAME "TensorT() not supported"); \
  }                                                                  \

TENSOR_IMPLEMENT_LOGICAL(lt)
TENSOR_IMPLEMENT_LOGICAL(gt)
TENSOR_IMPLEMENT_LOGICAL(le)
TENSOR_IMPLEMENT_LOGICAL(ge)
TENSOR_IMPLEMENT_LOGICAL(eq)
TENSOR_IMPLEMENT_LOGICAL(ne)

#undef TENSOR_IMPLEMENT_LOGICAL


template<>
auto THSTensor<real>::abs(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::abs() not supported");
}

template<>
auto THSTensor<real>::sigmoid(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::sigmoid() not supported");
}

template<>
auto THSTensor<real>::log(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::log() not supported");
}

template<>
auto THSTensor<real>::log1p(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::log1p() not supported");
}

template<>
auto THSTensor<real>::exp(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::exp() not supported");
}

template<>
auto THSTensor<real>::cos(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::cos() not supported");
}

template<>
auto THSTensor<real>::acos(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::acos() not supported");
}

template<>
auto THSTensor<real>::cosh(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::cosh() not supported");
}

template<>
auto THSTensor<real>::sin(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::sin() not supported");
}

template<>
auto THSTensor<real>::asin(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::asin() not supported");
}

template<>
auto THSTensor<real>::sinh(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::sinh() not supported");
}

template<>
auto THSTensor<real>::copy(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::copy() not supported");
}

template<>
auto THSTensor<real>::cat(const std::vector<Tensor*>& src, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::cat() not supported");
}

template<>
auto THSTensor<real>::gather(const Tensor& src, int dimension, const Tensor& index) -> THSTensor& {
  throw std::runtime_error("THSTensor::gather() not supported");
}

template<>
auto THSTensor<real>::scatter(int dimension, const Tensor& index, const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::scatter() not supported");
}

template<>
auto THSTensor<real>::scatterFill(int dimension, const Tensor& index, scalar_type value) -> THSTensor& {
  throw std::runtime_error("THSTensor::scatterFill() not supported");
}

template<>
auto THSTensor<real>::dot(const Tensor &src) -> scalar_type {
  throw std::runtime_error("THSTensor::dot() not supported");
}

template<>
auto THSTensor<real>::minall() -> scalar_type {
  throw std::runtime_error("THSTensor::minall() not supported");
}

template<>
auto THSTensor<real>::maxall() -> scalar_type {
  throw std::runtime_error("THSTensor::maxall() not supported");
}

template<>
auto THSTensor<real>::sumall() -> scalar_type {
  throw std::runtime_error("THSTensor::sumall() not supported");
}

template<>
auto THSTensor<real>::prodall() -> scalar_type {
  throw std::runtime_error("THSTensor::prodall() not supported");
}

template<>
auto THSTensor<real>::neg(const Tensor &src) -> THSTensor& {
  throw std::runtime_error("THSTensor::neg() not supported");
}

template<>
auto THSTensor<real>::cinv(const Tensor &src) -> THSTensor& {
  throw std::runtime_error("THSTensor::cinv() not supported");
}

template<>
auto THSTensor<real>::add(const Tensor &src, scalar_type value) -> THSTensor& {
  throw std::runtime_error("THSTensor::add() not supported");
}

template<>
auto THSTensor<real>::sub(const Tensor &src, scalar_type value) -> THSTensor& {
  throw std::runtime_error("THSTensor::sub() not supported");
}

template<>
auto THSTensor<real>::mul(const Tensor &src, scalar_type value) -> THSTensor& {
  const THSTensor &src_t = const_tensor_cast(src);
  THSTensor_(mul)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THSTensor<real>::div(const Tensor &src, scalar_type value) -> THSTensor& {
  const THSTensor &src_t = const_tensor_cast(src);
  THSTensor_(div)(tensor, src_t.tensor, value);
  return *this;
}

template<>
auto THSTensor<real>::fmod(const Tensor &src, scalar_type value) -> THSTensor& {
  throw std::runtime_error("THSTensor::fmod() not supported");
}

template<>
auto THSTensor<real>::remainder(const Tensor &src, scalar_type value) -> THSTensor& {
  throw std::runtime_error("THSTensor::remainder() not supported");
}

template<>
auto THSTensor<real>::clamp(const Tensor &src, scalar_type min_value, scalar_type max_value) -> THSTensor& {
  throw std::runtime_error("THSTensor::clamp() not supported");
}

template<>
auto THSTensor<real>::cadd(const Tensor& src1, scalar_type value, const Tensor& src2) -> THSTensor& {
  const THSTensor &src1_t = const_tensor_cast(src1);
  const THSTensor &src2_t = const_tensor_cast(src2);
  THSTensor_(cadd)(tensor, src1_t.tensor, value, src2_t.tensor);
  return *this;
}

template<>
auto THSTensor<real>::cadd(const Tensor& src1, const Tensor& src2) -> THSTensor& {
  return cadd(src1, static_cast<scalar_type>(1), src2);
}

template<>
auto THSTensor<real>::csub(const Tensor& src1, scalar_type value, const Tensor& src2) -> THSTensor& {
  throw std::runtime_error("THSTensor::csub() not supported");
}

template<>
auto THSTensor<real>::cmul(const Tensor& src1, const Tensor& src2) -> THSTensor& {
  const THSTensor &src1_t = const_tensor_cast(src1);
  const THSTensor &src2_t = const_tensor_cast(src2);
  THSTensor_(cmul)(tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THSTensor<real>::cpow(const Tensor& src1, const Tensor& src2) -> THSTensor& {
  throw std::runtime_error("THSTensor::cpow() not supported");
}

template<>
auto THSTensor<real>::cdiv(const Tensor& src1, const Tensor& src2) -> THSTensor& {
  throw std::runtime_error("THSTensor::cdiv() not supported");
}

template<>
auto THSTensor<real>::cfmod(const Tensor& src1, const Tensor& src2) -> THSTensor& {
  throw std::runtime_error("THSTensor::cfmod() not supported");
}

template<>
auto THSTensor<real>::cremainder(const Tensor& src1, const Tensor& src2) -> THSTensor& {
  throw std::runtime_error("THSTensor::cremainder() not supported");
}

template<>
auto THSTensor<real>::addcmul(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) -> THSTensor& {
  throw std::runtime_error("THSTensor::addcmul() not supported");
}

template<>
auto THSTensor<real>::addcdiv(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) -> THSTensor& {
  throw std::runtime_error("THSTensor::addcdiv() not supported");
}

template<>
auto THSTensor<real>::addmv(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat, const Tensor& vec) -> THSTensor& {
  throw std::runtime_error("THSTensor::addmv() not supported");
}

template<>
auto THSTensor<real>::addmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat1, const Tensor& mat2) -> THSTensor& {
  throw std::runtime_error("THSTensor::addmm() not supported");
}

template<>
auto THSTensor<real>::addr(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& vec1, const Tensor& vec2) -> THSTensor& {
  throw std::runtime_error("THSTensor::addr() not supported");
}

template<>
auto THSTensor<real>::addbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) -> THSTensor& {
  throw std::runtime_error("THSTensor::addbmm() not supported");
}

template<>
auto THSTensor<real>::baddbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) -> THSTensor& {
  throw std::runtime_error("THSTensor::baddbmm() not supported");
}

template<>
auto THSTensor<real>::match(const Tensor& m1, const Tensor& m2, scalar_type gain) -> THSTensor& {
  throw std::runtime_error("THSTensor::match() not supported");
}

template<>
auto THSTensor<real>::max(const Tensor& indices_, const Tensor& src, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::max() not supported");
}

template<>
auto THSTensor<real>::min(const Tensor& indices_, const Tensor& src, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::min() not supported");
}

template<>
auto THSTensor<real>::kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::kthvalue() not supported");
}

template<>
auto THSTensor<real>::mode(const Tensor& indices_, const Tensor& src, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::mode() not supported");
}

template<>
auto THSTensor<real>::median(const Tensor& indices_, const Tensor& src, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::median() not supported");
}

template<>
auto THSTensor<real>::sum(const Tensor& src, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::sum() not supported");
}

template<>
auto THSTensor<real>::prod(const Tensor& src, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::prod() not supported");
}

template<>
auto THSTensor<real>::cumsum(const Tensor& src, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::cumsum() not supported");
}

template<>
auto THSTensor<real>::cumprod(const Tensor& src, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::cumprod() not supported");
}

template<>
auto THSTensor<real>::sign(const Tensor& src) -> THSTensor& {
  throw std::runtime_error("THSTensor::sign() not supported");
}

template<>
auto THSTensor<real>::trace() -> scalar_type {
  throw std::runtime_error("THSTensor::trace() not supported");
}

template<>
auto THSTensor<real>::cross(const Tensor& src1, const Tensor& src2, int dimension) -> THSTensor& {
  throw std::runtime_error("THSTensor::cross() not supported");
}

template<>
auto THSTensor<real>::cmax(const Tensor& src1, const Tensor& src2) -> THSTensor& {
  throw std::runtime_error("THSTensor::cmax() not supported");
}

template<>
auto THSTensor<real>::cmin(const Tensor& src1, const Tensor& src2) -> THSTensor& {
  throw std::runtime_error("THSTensor::cmin() not supported");
}

template<>
auto THSTensor<real>::cmaxValue(const Tensor& src, scalar_type value) -> THSTensor& {
  throw std::runtime_error("THSTensor::cmaxValue() not supported");
}

template<>
auto THSTensor<real>::cminValue(const Tensor& src, scalar_type value) -> THSTensor& {
  throw std::runtime_error("THSTensor::cminValue() not supported");
}

template<>
auto THSTensor<real>::zero() -> THSTensor& {
  throw std::runtime_error("THSTensor::zero() not supported");
}

template<>
thpp::Type THSTensor<real>::type() const {
  return thpp::type_traits<real>::type;
}

template<>
bool THSTensor<real>::isCuda() const {
  return false;
}

template<>
bool THSTensor<real>::isSparse() const {
  return true;
}

template<>
int THSTensor<real>::getDevice() const {
  return -1;
}

template<>
std::unique_ptr<Tensor> THSTensor<real>::newTensor() const {
  return std::unique_ptr<Tensor>(new THSTensor<real>());
}

#endif

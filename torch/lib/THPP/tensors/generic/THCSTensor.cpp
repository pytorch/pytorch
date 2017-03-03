#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "tensors/generic/THCSTensor.cpp"
#else

#define const_tensor_cast(tensor) \
  dynamic_cast<const THCSTensor&>(tensor)
#define const_storage_cast(storage) \
  dynamic_cast<const THCStorage<real>&>(storage)
#define const_long_cast(tensor) \
  dynamic_cast<const THCSTensor<long>&>(tensor)
#define const_byte_cast(tensor) \
  dynamic_cast<const THCSTensor<unsigned char>&>(tensor)


template<>
THCSTensor<real>::THCSTensor(THCState* state):
  tensor(THCSTensor_(new)(state)), state(state)
  {};

template<>
THCSTensor<real>::THCSTensor(THCState* state, THCSRealTensor *wrapped):
  tensor(wrapped), state(state)
  {};

template<>
THCSTensor<real>::~THCSTensor() {
  if (tensor)
    THCSTensor_(free)(state, tensor);
}

template<>
auto THCSTensor<real>::clone() const -> THCSTensor* {
  return new THCSTensor(state, THCSTensor_(newClone)(state, tensor));
}

template<>
auto THCSTensor<real>::clone_shallow() -> THCSTensor* {
  THCSTensor_(retain)(state, tensor);
  return new THCSTensor(state, tensor);
}

template<>
auto THCSTensor<real>::contiguous() const -> std::unique_ptr<Tensor> {
  return std::unique_ptr<Tensor>(new THCSTensor(state, THCSTensor_(newContiguous)(state, tensor)));
}

template<>
int THCSTensor<real>::nDim() const {
  return tensor->nDimensionI;
}

template<>
auto THCSTensor<real>::sizes() const -> long_range {
  return std::vector<long>(tensor->size, tensor->size + tensor->nDimensionI);
}

template<>
const long* THCSTensor<real>::rawSizes() const {
  return tensor->size;
}

template<>
auto THCSTensor<real>::strides() const -> long_range {
  throw std::runtime_error("THCSTensor::strides() not supported");
}

template<>
const long* THCSTensor<real>::rawStrides() const {
  throw std::runtime_error("THCSTensor::rawStrides() not supported");
}

template<>
std::size_t THCSTensor<real>::storageOffset() const {
  throw std::runtime_error("THCSTensor::storageOffset() not supported");
}

template<>
std::size_t THCSTensor<real>::elementSize() const {
  return sizeof(real);
}

template<>
long long THCSTensor<real>::numel() const {
  throw std::runtime_error("THCSTensor::numel not supported");
}

template<>
bool THCSTensor<real>::isContiguous() const {
  throw std::runtime_error("THCSTensor::isContiguous() not supported");
}

template<>
void* THCSTensor<real>::data() {
  throw std::runtime_error("THCSTensor::data() not supported");
}

template<>
const void* THCSTensor<real>::data() const {
  throw std::runtime_error("THCSTensor::data() not supported");
}

template<>
void* THCSTensor<real>::cdata() {
  return tensor;
}

template<>
const void* THCSTensor<real>::cdata() const {
  return tensor;
}

template<>
auto THCSTensor<real>::resize(const std::initializer_list<long> &new_size) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::resize() not supported");
}

template<>
auto THCSTensor<real>::resize(const std::vector<long> &new_size) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::resize() not supported");
}

template<>
auto THCSTensor<real>::resize(THLongStorage *size, THLongStorage *stride) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::resize() not supported");
}

template<>
auto THCSTensor<real>::resizeAs(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::resizeAs() not supported");
}

template<>
template<typename iterator>
auto THCSTensor<real>::resize(const iterator& begin, const iterator& end) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::resize() not supported");
}

template<>
auto THCSTensor<real>::set(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::set() not supported");
}

template<>
auto THCSTensor<real>::setStorage(const Storage& storage,
                                 ptrdiff_t storageOffset,
                                 const long_range& size,
                                 const long_range& stride) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::setStorage not supported");
}

template<>
auto THCSTensor<real>::setStorage(const Storage& storage,
                                 ptrdiff_t storageOffset,
                                 THLongStorage *size,
                                 THLongStorage *stride) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::setStorage not supported");
}

template<>
auto THCSTensor<real>::narrow(const Tensor& src,
                             int dimension,
                             long firstIndex,
                             long size) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::narrow not supported");
}

template<>
auto THCSTensor<real>::select(const Tensor& src, int dimension,
                             long sliceIndex) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::select not supported");
}

template<>
auto THCSTensor<real>::transpose(const Tensor& src, int dimension1,
                                int dimension2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::transpose not supported");
}


template<>
auto THCSTensor<real>::unfold(const Tensor& src, int dimension,
                             long size, long step) ->THCSTensor& {
  throw std::runtime_error("THCSTensor::unfold not supported");
}

template<>
auto THCSTensor<real>::squeeze(const Tensor& src, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::squeeze not supported");
}

template<>
auto THCSTensor<real>::unsqueeze(const Tensor& src, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::unsqueeze not supported");
}

#ifdef THCS_REAL_IS_HALF
#define cast_scalar(v) THC_float2half(v)
#define uncast_scalar(v) THC_half2float(v)
#else
#define cast_scalar(v) v
#define uncast_scalar(v) v
#endif

template<>
auto THCSTensor<real>::fill(scalar_type value) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::fill() not supported");
}

template<>
auto THCSTensor<real>::retain() -> THCSTensor& {
  THCSTensor_(retain)(state, tensor);
  return *this;
}

template<>
auto THCSTensor<real>::free() -> THCSTensor& {
  THCSTensor_(free)(state, tensor);
  return *this;
}

template<>
auto THCSTensor<real>::diag(const Tensor& src, int k) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::diag() not supported");
}

template<>
auto THCSTensor<real>::eye(long n, long m) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::eye() not supported");
}

template<>
auto THCSTensor<real>::range(scalar_type xmin, scalar_type xmax,
                           scalar_type step) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::range() not supported");
}

template<>
auto THCSTensor<real>::sort(const Tensor& ri, const Tensor& src,
                          int dimension, int desc) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::sort() not supported");
}

template<>
auto THCSTensor<real>::topk(const Tensor& ri, const Tensor& src,
                          long k, int dim, int dir, int sorted) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::topk() not supported");
}

template<>
auto THCSTensor<real>::tril(const Tensor& src, long k) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::tril() not supported");
}

template<>
auto THCSTensor<real>::triu(const Tensor& src, long k) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::triu() not supported");
}

template<>
auto THCSTensor<real>::catArray(const std::vector<Tensor*>& inputs_vec,
                              int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::catArray() not supported");
}

template<>
int THCSTensor<real>::equal(const Tensor& other) const {
  throw std::runtime_error("THCSTensor::equal() not supported");
}

  // Note: the order in *Value and *Tensor is reversed compared to
  // the declarations in TH/generic/THCSTensorMath.h, so for instance
  // the call THRealTensor_ltTensor(r, ta, tb) is equivalent to
  // ta->ltTensor(r, tb). It is done this way so that the first
  // argument can be casted onto a byte tensor type

#define TENSOR_IMPLEMENT_LOGICAL(NAME)                               \
  template<>                                                         \
  auto THCSTensor<real>::NAME##Value(const Tensor& r,                \
                                     scalar_type value) -> THCSTensor& { \
    throw std::invalid_argument("THCSTensor::" #NAME "Value() not supported"); \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THCSTensor<real>::NAME##ValueT(const Tensor& t,               \
                                      scalar_type value) -> THCSTensor& { \
    throw std::invalid_argument("THCSTensor::" #NAME "ValueT() not supported"); \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THCSTensor<real>::NAME##Tensor(const Tensor& r,               \
                                      const Tensor& tb) -> THCSTensor& { \
    throw std::invalid_argument("THCSTensor::" #NAME "Tensor() not supported"); \
  }                                                                  \
                                                                     \
  template<>                                                         \
  auto THCSTensor<real>::NAME##TensorT(const Tensor& ta,             \
                                       const Tensor& tb) -> THCSTensor& { \
    throw std::invalid_argument("THCSTensor::" #NAME "TensorT() not supported"); \
  }                                                                  \

TENSOR_IMPLEMENT_LOGICAL(lt)
TENSOR_IMPLEMENT_LOGICAL(gt)
TENSOR_IMPLEMENT_LOGICAL(le)
TENSOR_IMPLEMENT_LOGICAL(ge)
TENSOR_IMPLEMENT_LOGICAL(eq)
TENSOR_IMPLEMENT_LOGICAL(ne)

#undef TENSOR_IMPLEMENT_LOGICAL


template<>
auto THCSTensor<real>::abs(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::abs() not supported");
}

template<>
auto THCSTensor<real>::sigmoid(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::sigmoid() not supported");
}

template<>
auto THCSTensor<real>::log(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::log() not supported");
}

template<>
auto THCSTensor<real>::log1p(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::log1p() not supported");
}

template<>
auto THCSTensor<real>::exp(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::exp() not supported");
}

template<>
auto THCSTensor<real>::cos(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cos() not supported");
}

template<>
auto THCSTensor<real>::acos(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::acos() not supported");
}

template<>
auto THCSTensor<real>::cosh(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cosh() not supported");
}

template<>
auto THCSTensor<real>::sin(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::sin() not supported");
}

template<>
auto THCSTensor<real>::asin(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::asin() not supported");
}

template<>
auto THCSTensor<real>::sinh(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::sinh() not supported");
}

template<>
auto THCSTensor<real>::copy(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::copy() not supported");
}

template<>
auto THCSTensor<real>::cat(const std::vector<Tensor*>& src, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cat() not supported");
}

template<>
auto THCSTensor<real>::gather(const Tensor& src, int dimension, const Tensor& index) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::gather() not supported");
}

template<>
auto THCSTensor<real>::scatter(int dimension, const Tensor& index, const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::scatter() not supported");
}

template<>
auto THCSTensor<real>::scatterFill(int dimension, const Tensor& index, scalar_type value) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::scatterFill() not supported");
}

template<>
auto THCSTensor<real>::dot(const Tensor &src) -> scalar_type {
  throw std::runtime_error("THCSTensor::dot() not supported");
}

template<>
auto THCSTensor<real>::minall() -> scalar_type {
  throw std::runtime_error("THCSTensor::minall() not supported");
}

template<>
auto THCSTensor<real>::maxall() -> scalar_type {
  throw std::runtime_error("THCSTensor::maxall() not supported");
}

template<>
auto THCSTensor<real>::sumall() -> scalar_type {
  throw std::runtime_error("THCSTensor::sumall() not supported");
}

template<>
auto THCSTensor<real>::prodall() -> scalar_type {
  throw std::runtime_error("THCSTensor::prodall() not supported");
}

template<>
auto THCSTensor<real>::neg(const Tensor &src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::neg() not supported");
}

template<>
auto THCSTensor<real>::cinv(const Tensor &src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cinv() not supported");
}

template<>
auto THCSTensor<real>::add(const Tensor &src, scalar_type value) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::add() not supported");
}

template<>
auto THCSTensor<real>::sub(const Tensor &src, scalar_type value) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::sub() not supported");
}

template<>
auto THCSTensor<real>::mul(const Tensor &src, scalar_type value) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::mul() not supported");
}

template<>
auto THCSTensor<real>::div(const Tensor &src, scalar_type value) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::div() not supported");
}

template<>
auto THCSTensor<real>::fmod(const Tensor &src, scalar_type value) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::fmod() not supported");
}

template<>
auto THCSTensor<real>::remainder(const Tensor &src, scalar_type value) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::remainder() not supported");
}

template<>
auto THCSTensor<real>::clamp(const Tensor &src, scalar_type min_value, scalar_type max_value) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::clamp() not supported");
}

template<>
auto THCSTensor<real>::cadd(const Tensor& src1, scalar_type value, const Tensor& src2) -> THCSTensor& {
  const THCSTensor &src1_t = const_tensor_cast(src1);
  const THCSTensor &src2_t = const_tensor_cast(src2);
  THCSTensor_(cadd)(state, tensor, src1_t.tensor, cast_scalar(value), src2_t.tensor);
  return *this;
}

template<>
auto THCSTensor<real>::cadd(const Tensor& src1, const Tensor& src2) -> THCSTensor& {
  return cadd(src1, static_cast<scalar_type>(1), src2);
  throw std::runtime_error("THCSTensor::cadd() not supported");
}

template<>
auto THCSTensor<real>::csub(const Tensor& src1, scalar_type value, const Tensor& src2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::csub() not supported");
}

template<>
auto THCSTensor<real>::cmul(const Tensor& src1, const Tensor& src2) -> THCSTensor& {
  const THCSTensor &src1_t = const_tensor_cast(src1);
  const THCSTensor &src2_t = const_tensor_cast(src2);
  THCSTensor_(cmul)(state, tensor, src1_t.tensor, src2_t.tensor);
  return *this;
}

template<>
auto THCSTensor<real>::cpow(const Tensor& src1, const Tensor& src2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cpow() not supported");
}

template<>
auto THCSTensor<real>::cdiv(const Tensor& src1, const Tensor& src2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cdiv() not supported");
}

template<>
auto THCSTensor<real>::cfmod(const Tensor& src1, const Tensor& src2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cfmod() not supported");
}

template<>
auto THCSTensor<real>::cremainder(const Tensor& src1, const Tensor& src2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cremainder() not supported");
}

template<>
auto THCSTensor<real>::addcmul(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::addcmul() not supported");
}

template<>
auto THCSTensor<real>::addcdiv(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::addcdiv() not supported");
}

template<>
auto THCSTensor<real>::addmv(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat, const Tensor& vec) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::addmv() not supported");
}

template<>
auto THCSTensor<real>::addmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat1, const Tensor& mat2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::addmm() not supported");
}

template<>
auto THCSTensor<real>::addr(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& vec1, const Tensor& vec2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::addr() not supported");
}

template<>
auto THCSTensor<real>::addbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::addbmm() not supported");
}

template<>
auto THCSTensor<real>::baddbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::baddbmm() not supported");
}

template<>
auto THCSTensor<real>::match(const Tensor& m1, const Tensor& m2, scalar_type gain) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::match() not supported");
}

template<>
auto THCSTensor<real>::max(const Tensor& indices_, const Tensor& src, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::max() not supported");
}

template<>
auto THCSTensor<real>::min(const Tensor& indices_, const Tensor& src, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::min() not supported");
}

template<>
auto THCSTensor<real>::kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::kthvalue() not supported");
}

template<>
auto THCSTensor<real>::mode(const Tensor& indices_, const Tensor& src, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::mode() not supported");
}

template<>
auto THCSTensor<real>::median(const Tensor& indices_, const Tensor& src, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::median() not supported");
}

template<>
auto THCSTensor<real>::sum(const Tensor& src, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::sum() not supported");
}

template<>
auto THCSTensor<real>::prod(const Tensor& src, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::prod() not supported");
}

template<>
auto THCSTensor<real>::cumsum(const Tensor& src, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cumsum() not supported");
}

template<>
auto THCSTensor<real>::cumprod(const Tensor& src, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cumprod() not supported");
}

template<>
auto THCSTensor<real>::sign(const Tensor& src) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::sign() not supported");
}

template<>
auto THCSTensor<real>::trace() -> scalar_type {
  throw std::runtime_error("THCSTensor::trace() not supported");
}

template<>
auto THCSTensor<real>::cross(const Tensor& src1, const Tensor& src2, int dimension) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cross() not supported");
}

template<>
auto THCSTensor<real>::cmax(const Tensor& src1, const Tensor& src2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cmax() not supported");
}

template<>
auto THCSTensor<real>::cmin(const Tensor& src1, const Tensor& src2) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cmin() not supported");
}

template<>
auto THCSTensor<real>::cmaxValue(const Tensor& src, scalar_type value) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cmaxValue() not supported");
}

template<>
auto THCSTensor<real>::cminValue(const Tensor& src, scalar_type value) -> THCSTensor& {
  throw std::runtime_error("THCSTensor::cminValue() not supported");
}

template<>
auto THCSTensor<real>::zero() -> THCSTensor& {
  throw std::runtime_error("THCSTensor::zero() not supported");
}

template<>
thpp::Type THCSTensor<real>::type() const {
  return thpp::type_traits<real>::type;
}

template<>
bool THCSTensor<real>::isCuda() const {
  return false;
}

template<>
bool THCSTensor<real>::isSparse() const {
  return true;
}

template<>
int THCSTensor<real>::getDevice() const {
  return -1;
}

template<>
std::unique_ptr<Tensor> THCSTensor<real>::newTensor() const {
  return std::unique_ptr<Tensor>(new THCSTensor<real>(state));
}

#undef cast_scalar
#undef uncast_scalar

#endif

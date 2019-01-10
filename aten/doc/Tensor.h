#pragma once

#include "ATen/Generator.h"
#include "ATen/Scalar.h"
#include "ATen/ScalarType.h"
#include "ATen/TensorAccessor.h"
#include "ATen/TensorImpl.h"
#include "ATen/TensorBase.h"
#include "ATen/Storage.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/Utils.h"

namespace at {
struct Type;

// Tensor is a "generic" object holding a pointer to the underlying TensorImpl object, which
// has an embedded reference count. In this way, Tensor is similar to boost::intrusive_ptr.
//
// For example:
//
// void func(Tensor a) {
//   Tensor b = a;
//   ...
// }
//
// In this example, when we say Tensor b = a, we are creating a new object that points to the
// same underlying TensorImpl, and bumps its reference count. When b goes out of scope, the
// destructor decrements the reference count by calling release() on the TensorImpl it points to.
// The existing constructors, operator overloads, etc. take care to implement the correct semantics.
//
// Note that Tensor can also be NULL, i.e. it is not associated with any underlying TensorImpl, and
// special care must be taken to handle this.
struct Tensor : public detail::TensorBase {
  Tensor() : TensorBase() {}
  Tensor(TensorImpl * self, bool retain) : TensorBase(self, retain) {}
  Tensor(const TensorBase & rhs) : TensorBase(rhs) {}
  Tensor(const Tensor & rhs) = default;
  Tensor(Tensor && rhs) noexcept = default;

  // reimplemented from TensorBase so the return type is Tensor rather than TensorBase
  Tensor & operator=(Tensor && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  Tensor & operator=(Tensor const & rhs) & {
      //Tensor ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally Tensor dtor releases rhs.pImpl, which was originally this->pImpl
      Tensor(rhs).swap(*this);
      return *this;
  }

  inline Tensor & operator=(Tensor const & rhs) &&;
  Tensor & operator=(Scalar v) &&;
  const char * toString() const {
    return pImpl->toString();
  }
  IntList sizes() const {
    return pImpl->sizes();
  }
  IntList strides() const {
    return pImpl->strides();
  }
  int64_t ndimension() const {
    return dim();
  }
  Type & type() const {
    return pImpl->type();
  }
  std::unique_ptr<Storage> storage() const {
    return pImpl->storage();
  }
  inline Tensor toType(const Type & t) const;
  inline Tensor & copy_(const Tensor & src, bool non_blocking=false);
  inline Tensor toType(ScalarType t) const;
  inline Tensor toBackend(Backend b) const;

  template<typename T>
  T * data() const;

  void * unsafeGetTH(bool retain) const {
    return pImpl->unsafeGetTH(retain);
  }

  // Purposely not defined here to avoid inlining
  void print() const;

  //toLongData(), toFloatData() etc.
  #define TO_TYPE_DATA(T,name,_) \
  T * to##name##Data() const;
  AT_FORALL_SCALAR_TYPES(TO_TYPE_DATA)
  #undef TO_TYPE_DATA

  #define TO_C_TYPE(T,name,_) \
  T toC##name () const;
  AT_FORALL_SCALAR_TYPES(TO_C_TYPE)
  #undef TO_C_TYPE

  template<typename T, size_t N>
  TensorAccessor<T,N> accessor() {
    static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data<T>()");
    AT_ASSERT(dim() == N, "expected %d dims but tensor has %d",N,dim());
    return TensorAccessor<T,N>(data<T>(),sizes().data(),strides().data());
  }

  Tensor operator-() const;
  Tensor& operator+=(const Tensor & other);
  Tensor& operator+=(Scalar other);
  Tensor& operator-=(const Tensor & other);
  Tensor& operator-=(Scalar other);
  Tensor& operator*=(const Tensor & other);
  Tensor& operator*=(Scalar other);
  Tensor& operator/=(const Tensor & other);
  Tensor& operator/=(Scalar other);
  Tensor operator[](int64_t idx) const;

  // STOP.  Thinking of adding a method here, which only makes use
  // of other ATen methods?  Define it in native_functions.yaml.

  //example
  //Tensor * add(Tensor & b);
  int64_t storage_offset() const;
  Tensor & resize_(IntList size);
  int64_t numel() const;
  Tensor & set_(Storage & storage);
  Tensor & set_(Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride={});
  Tensor & set_(const Tensor & source);
  Tensor & set_();
  Tensor & fill_(Scalar value);
  Tensor & fill_(const Tensor & value);
  bool is_contiguous() const;
  bool is_set_to(const Tensor & tensor) const;
  Tensor & masked_fill_(const Tensor & mask, Scalar value);
  Tensor & masked_fill_(const Tensor & mask, const Tensor & value);
  Tensor & masked_scatter_(const Tensor & mask, const Tensor & source);
  Tensor masked_select(const Tensor & mask) const;
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  Tensor t() const;
  Tensor nonzero() const;
  Tensor contiguous() const;
  Tensor clone() const;
  Tensor view(IntList size) const;
  Tensor & resize_as_(const Tensor & the_template);
  Tensor index_select(int64_t dim, const Tensor & index) const;
  Tensor & index_copy_(int64_t dim, const Tensor & index, const Tensor & source);
  Tensor take(const Tensor & index) const;
  Tensor & put_(const Tensor & index, const Tensor & source, bool accumulate=false);
  Tensor & index_add_(int64_t dim, const Tensor & index, const Tensor & source);
  Tensor & index_fill_(int64_t dim, const Tensor & index, Scalar value);
  Tensor & index_fill_(int64_t dim, const Tensor & index, const Tensor & value);
  Tensor unfold(int64_t dimension, int64_t size, int64_t step) const;
  Tensor & scatter_(int64_t dim, const Tensor & index, const Tensor & src);
  Tensor & scatter_(int64_t dim, const Tensor & index, Scalar value);
  Tensor & scatter_add_(int64_t dim, const Tensor & index, const Tensor & src);
  Tensor gather(int64_t dim, const Tensor & index) const;
  void* data_ptr() const;
  bool equal(const Tensor & other) const;
  Tensor __and__(Scalar other) const;
  Tensor __and__(const Tensor & other) const;
  Tensor & __iand__(Scalar other);
  Tensor & __iand__(const Tensor & other);
  Tensor __or__(Scalar other) const;
  Tensor __or__(const Tensor & other) const;
  Tensor & __ior__(Scalar other);
  Tensor & __ior__(const Tensor & other);
  Tensor __xor__(Scalar other) const;
  Tensor __xor__(const Tensor & other) const;
  Tensor & __ixor__(Scalar other);
  Tensor & __ixor__(const Tensor & other);
  Tensor __lshift__(Scalar other) const;
  Tensor __lshift__(const Tensor & other) const;
  Tensor & __ilshift__(Scalar other);
  Tensor & __ilshift__(const Tensor & other);
  Tensor __rshift__(Scalar other) const;
  Tensor __rshift__(const Tensor & other) const;
  Tensor & __irshift__(Scalar other);
  Tensor & __irshift__(const Tensor & other);
  Tensor lt(Scalar other) const;
  Tensor lt(const Tensor & other) const;
  Tensor & lt_(Scalar other);
  Tensor & lt_(const Tensor & other);
  Tensor gt(Scalar other) const;
  Tensor gt(const Tensor & other) const;
  Tensor & gt_(Scalar other);
  Tensor & gt_(const Tensor & other);
  Tensor le(Scalar other) const;
  Tensor le(const Tensor & other) const;
  Tensor & le_(Scalar other);
  Tensor & le_(const Tensor & other);
  Tensor ge(Scalar other) const;
  Tensor ge(const Tensor & other) const;
  Tensor & ge_(Scalar other);
  Tensor & ge_(const Tensor & other);
  Tensor eq(Scalar other) const;
  Tensor eq(const Tensor & other) const;
  Tensor & eq_(Scalar other);
  Tensor & eq_(const Tensor & other);
  Tensor ne(Scalar other) const;
  Tensor ne(const Tensor & other) const;
  Tensor & ne_(Scalar other);
  Tensor & ne_(const Tensor & other);
  std::tuple<Tensor,Tensor> min(int64_t dim, bool keepdim=false) const;
  Tensor min(const Tensor & other) const;
  Tensor min() const;
  std::tuple<Tensor,Tensor> max(int64_t dim, bool keepdim=false) const;
  Tensor max(const Tensor & other) const;
  Tensor max() const;
  std::tuple<Tensor,Tensor> kthvalue(int64_t k, int64_t dim=-1, bool keepdim=false) const;
  std::tuple<Tensor,Tensor> mode(int64_t dim=-1, bool keepdim=false) const;
  std::tuple<Tensor,Tensor> median(int64_t dim, bool keepdim=false) const;
  Tensor median() const;
  std::tuple<Tensor,Tensor> sort(int64_t dim=-1, bool descending=false) const;
  std::tuple<Tensor,Tensor> topk(int64_t k, int64_t dim=-1, bool largest=true, bool sorted=true) const;
  Tensor all() const;
  Tensor any() const;
  int64_t get_device() const;
  Tensor abs() const;
  Tensor & abs_();
  Tensor & sigmoid_();
  Tensor sigmoid() const;
  Tensor & log_();
  Tensor log() const;
  Tensor & log1p_();
  Tensor log1p() const;
  Tensor lgamma() const;
  Tensor & lgamma_();
  Tensor digamma() const;
  Tensor & digamma_();
  Tensor polygamma(int64_t n) const;
  Tensor & polygamma_(int64_t n);
  Tensor & exp_();
  Tensor exp() const;
  Tensor & expm1_();
  Tensor expm1() const;
  Tensor & cos_();
  Tensor cos() const;
  Tensor & acos_();
  Tensor acos() const;
  Tensor & cosh_();
  Tensor cosh() const;
  Tensor & sin_();
  Tensor sin() const;
  Tensor & asin_();
  Tensor asin() const;
  Tensor & sinh_();
  Tensor sinh() const;
  Tensor & tan_();
  Tensor tan() const;
  Tensor & atan_();
  Tensor atan() const;
  Tensor & tanh_();
  Tensor tanh() const;
  Tensor & erf_();
  Tensor erf() const;
  Tensor & erfinv_();
  Tensor erfinv() const;
  Tensor & sqrt_();
  Tensor sqrt() const;
  Tensor & rsqrt_();
  Tensor rsqrt() const;
  Tensor & ceil_();
  Tensor ceil() const;
  Tensor & floor_();
  Tensor floor() const;
  Tensor & round_();
  Tensor round() const;
  Tensor & trunc_();
  Tensor trunc() const;
  Tensor & frac_();
  Tensor frac() const;
  Tensor mean(int64_t dim, bool keepdim=false) const;
  Tensor mean() const;
  Tensor var(int64_t dim, bool unbiased=true, bool keepdim=false) const;
  Tensor var(bool unbiased=true) const;
  Tensor std(int64_t dim, bool unbiased=true, bool keepdim=false) const;
  Tensor std(bool unbiased=true) const;
  Tensor norm(Scalar p, int64_t dim, bool keepdim=false) const;
  Tensor norm(Scalar p=2) const;
  Tensor renorm(Scalar p, int64_t dim, Scalar maxnorm) const;
  Tensor & renorm_(Scalar p, int64_t dim, Scalar maxnorm);
  Tensor dist(const Tensor & other, Scalar p=2) const;
  Tensor reciprocal() const;
  Tensor & reciprocal_();
  Tensor neg() const;
  Tensor & neg_();
  Tensor atan2(const Tensor & other) const;
  Tensor & atan2_(const Tensor & other);
  Tensor pow(Scalar exponent) const;
  Tensor pow(const Tensor & exponent) const;
  Tensor & pow_(Scalar exponent);
  Tensor & pow_(const Tensor & exponent);
  Tensor lerp(const Tensor & end, Scalar weight) const;
  Tensor & lerp_(const Tensor & end, Scalar weight);
  Tensor histc(int64_t bins=100, Scalar min=0, Scalar max=0) const;
  Tensor & zero_();
  Tensor sum(int64_t dim, bool keepdim=false) const;
  Tensor sum() const;
  Tensor prod(int64_t dim, bool keepdim=false) const;
  Tensor prod() const;
  Tensor cumsum(int64_t dim) const;
  Tensor cumprod(int64_t dim) const;
  Tensor sign() const;
  Tensor & sign_();
  Tensor trace() const;
  Tensor add(Scalar other, Scalar alpha=1) const;
  Tensor add(const Tensor & other, Scalar alpha=1) const;
  Tensor add(SparseTensor other, Scalar alpha=1) const;
  Tensor & add_(Scalar other, Scalar alpha=1);
  Tensor & add_(const Tensor & other, Scalar alpha=1);
  Tensor & add_(SparseTensor other, Scalar alpha=1);
  Tensor sub(Scalar other, Scalar alpha=1) const;
  Tensor sub(const Tensor & other, Scalar alpha=1) const;
  Tensor & sub_(Scalar other, Scalar alpha=1);
  Tensor & sub_(const Tensor & other, Scalar alpha=1);
  Tensor mul(Scalar other) const;
  Tensor mul(const Tensor & other) const;
  Tensor & mul_(Scalar other);
  Tensor & mul_(const Tensor & other);
  Tensor div(Scalar other) const;
  Tensor div(const Tensor & other) const;
  Tensor & div_(Scalar other);
  Tensor & div_(const Tensor & other);
  Tensor fmod(Scalar other) const;
  Tensor fmod(const Tensor & other) const;
  Tensor & fmod_(Scalar other);
  Tensor & fmod_(const Tensor & other);
  Tensor remainder(Scalar other) const;
  Tensor remainder(const Tensor & other) const;
  Tensor & remainder_(Scalar other);
  Tensor & remainder_(const Tensor & other);
  Tensor clamp(Scalar min, Scalar max) const;
  Tensor & clamp_(Scalar min, Scalar max);
  Tensor clamp_min(Scalar min) const;
  Tensor & clamp_min_(Scalar min);
  Tensor clamp_max(Scalar max) const;
  Tensor & clamp_max_(Scalar max);
  Tensor _dot(const Tensor & tensor) const;
  Tensor tril(int64_t diagonal=0) const;
  Tensor & tril_(int64_t diagonal=0);
  Tensor triu(int64_t diagonal=0) const;
  Tensor & triu_(int64_t diagonal=0);
  Tensor cross(const Tensor & other, int64_t dim=-1) const;
  Tensor diag(int64_t diagonal=0) const;
  Tensor addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1) const;
  Tensor addmm(SparseTensor mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1) const;
  Tensor & addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
  Tensor & addmm_(SparseTensor mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
  Tensor _addmv(const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1) const;
  Tensor & _addmv_(const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
  Tensor _addr(const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1) const;
  Tensor & _addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
  Tensor _ger(const Tensor & vec2) const;
  Tensor _mv(const Tensor & vec) const;
  Tensor _mm(const Tensor & mat2) const;
  Tensor bmm(const Tensor & mat2) const;
  Tensor addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  Tensor & addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1);
  Tensor baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
  Tensor & baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1);
  Tensor addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  Tensor & addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value=1);
  Tensor addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
  Tensor & addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value=1);
  std::tuple<Tensor,Tensor> gesv(const Tensor & A) const;
  std::tuple<Tensor,Tensor> gels(const Tensor & A) const;
  std::tuple<Tensor,Tensor> trtrs(const Tensor & A, bool upper=true, bool transpose=false, bool unitriangular=false) const;
  std::tuple<Tensor,Tensor> symeig(bool eigenvectors=false, bool upper=true) const;
  std::tuple<Tensor,Tensor> eig(bool eigenvectors=false) const;
  std::tuple<Tensor,Tensor,Tensor> svd(bool some=true) const;
  Tensor inverse() const;
  Tensor potrf(bool upper=true) const;
  Tensor potrs(const Tensor & input2, bool upper=true) const;
  Tensor potri(bool upper=true) const;
  std::tuple<Tensor,Tensor> pstrf(bool upper=true, Scalar tol=-1) const;
  std::tuple<Tensor,Tensor> qr() const;
  std::tuple<Tensor,Tensor> geqrf() const;
  Tensor orgqr(const Tensor & input2) const;
  Tensor ormqr(const Tensor & input2, const Tensor & input3, bool left=true, bool transpose=false) const;
  std::tuple<Tensor,Tensor> btrifact(bool pivot=true) const;
  std::tuple<Tensor,Tensor,Tensor> btrifact_with_info(bool pivot=true) const;
  Tensor btrisolve(const Tensor & LU_data, const Tensor & LU_pivots) const;
  Tensor & random_(int64_t from, int64_t to, Generator * generator=nullptr);
  Tensor & random_(int64_t to, Generator * generator=nullptr);
  Tensor & random_(Generator * generator=nullptr);
  Tensor multinomial(int64_t num_samples, bool replacement=false, Generator * generator=nullptr) const;
  Tensor & uniform_(double from=0, double to=1, Generator * generator=nullptr);
  Tensor & normal_(double mean=0, double std=1, Generator * generator=nullptr);
  Tensor & cauchy_(double median=0, double sigma=1, Generator * generator=nullptr);
  Tensor & log_normal_(double mean=1, double std=2, Generator * generator=nullptr);
  Tensor & exponential_(double lambd=1, Generator * generator=nullptr);
  Tensor & geometric_(double p, Generator * generator=nullptr);
  Tensor bernoulli(Generator * generator=nullptr) const;
  Tensor _standard_gamma(Generator * generator=nullptr) const;
  Tensor & _copy_ignoring_overlaps_(const Tensor & src);
  Tensor as_strided(IntList size, IntList stride, int64_t storage_offset=-1) const;
  Tensor & as_strided_(IntList size, IntList stride, int64_t storage_offset=-1);
  Tensor & sparse_raw_resize_(IntList size, int64_t nDimI, int64_t nDimV);
  Tensor & reshape_(IntList size, IntList stride);
  Tensor _sparse_mask(SparseTensor mask) const;
  Tensor to_dense() const;
  int64_t _dimI() const;
  int64_t _dimV() const;
  int64_t _nnz() const;
  Tensor coalesce() const;
  bool is_coalesced() const;
  Tensor _indices() const;
  Tensor _values() const;
  bool allclose(const Tensor & other, double rtol=1e-05, double atol=1e-08) const;
  Tensor addmv(const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1) const;
  Tensor & addmv_(const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
  Tensor addr(const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1) const;
  Tensor & addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
  Tensor & bernoulli_(const Tensor & p, Generator * generator=nullptr);
  Tensor & bernoulli_(double p=0.5, Generator * generator=nullptr);
  Tensor sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1) const;
  std::vector<Tensor> chunk(int64_t chunks, int64_t dim=0) const;
  Tensor conv_tbc(const Tensor & weight, const Tensor & bias, int64_t pad) const;
  std::tuple<Tensor,Tensor,Tensor> conv_tbc_backward(const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) const;
  Tensor det() const;
  std::tuple<Tensor,Tensor,Tensor,Tensor> _det_with_svd() const;
  Tensor dot(const Tensor & tensor) const;
  Tensor expand(IntList size) const;
  Tensor expand_as(const Tensor & other) const;
  Tensor ger(const Tensor & vec2) const;
  Tensor index(TensorList indices) const;
  Tensor & index_put_(TensorList indices, const Tensor & values);
  bool is_cuda() const;
  bool is_distributed() const;
  bool is_floating_point() const;
  bool is_nonzero() const;
  bool is_same_size(const Tensor & other) const;
  bool is_signed() const;
  bool is_sparse() const;
  Tensor matmul(const Tensor & other) const;
  Tensor mm(const Tensor & mat2) const;
  Tensor mv(const Tensor & vec) const;
  Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
  Tensor permute(IntList dims) const;
  Tensor pin_memory() const;
  Tensor repeat(IntList repeats) const;
  Tensor select(int64_t dim, int64_t index) const;
  int64_t size(int64_t dim) const;
  Tensor slice(int64_t dim=0, int64_t start=0, int64_t end=9223372036854775807, int64_t step=1) const;
  std::vector<Tensor> split(int64_t split_size, int64_t dim=0) const;
  Tensor squeeze() const;
  Tensor squeeze(int64_t dim) const;
  Tensor & squeeze_();
  Tensor & squeeze_(int64_t dim);
  Tensor stft(int64_t frame_length, int64_t hop, int64_t fft_size, bool return_onesided=true, const Tensor & window={}, int64_t pad_end=0) const;
  int64_t stride(int64_t dim) const;
  Tensor & transpose_(int64_t dim0, int64_t dim1);
  Tensor & t_();
  Tensor type_as(const Tensor & other) const;
  Tensor unsqueeze(int64_t dim) const;
  Tensor & unsqueeze_(int64_t dim);
  Tensor view_as(const Tensor & other) const;
  Tensor where(const Tensor & condition, const Tensor & other) const;
  Tensor _s_where(const Tensor & condition, const Tensor & other) const;
  Tensor _standard_gamma_grad(const Tensor & output) const;
};

} //namespace at

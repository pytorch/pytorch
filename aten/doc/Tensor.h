#pragma once

#include "ATen/Type.h"
#include "ATen/TensorImpl.h"
#include "ATen/Utils.h"
#include "ATen/TensorAccessor.h"

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
struct Tensor {

  Tensor()
  : pImpl(nullptr){}
  explicit Tensor(TensorImpl * self, bool retain)
  : pImpl(self) {
    if(pImpl != nullptr && retain)
      pImpl->retain();
  }
  Tensor(Tensor const & rhs)
  : pImpl(rhs.pImpl) {
    if(pImpl != nullptr)
      pImpl->retain();
  }
  Tensor(Tensor && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = nullptr;
  }
  ~Tensor() {
    if(pImpl != nullptr)
      pImpl->release();
  }
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
  Tensor & operator=(Tensor const & rhs) && {
    return assign_(rhs);
  }
  Tensor & operator=(Scalar v) &&;
  Tensor & assign_(Scalar v);
  void reset() {
    Tensor().swap(*this);
  }
  void reset(TensorImpl * rhs) {
    Tensor(rhs,true).swap(*this);
  }
  void reset(TensorImpl * rhs, bool retain) {
    Tensor(rhs, retain).swap(*this );
  }
  TensorImpl * get() const {
    return pImpl;
  }
  TensorImpl * detach() {
    TensorImpl * ret = pImpl;
    pImpl = nullptr;
    return ret;
  }
  bool defined() const {
    return pImpl != nullptr;
  }
  void swap(Tensor & rhs) {
    TensorImpl * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }
  const char * toString() const {
    return pImpl->toString();
  }
  IntList sizes() const {
    return pImpl->sizes();
  }
  IntList strides() const {
    return pImpl->strides();
  }
  int64_t dim() const {
    return pImpl->dim();
  }
  int64_t ndimension() const {
    return dim();
  }
  Type & type() const {
    return pImpl->type();
  }
  Tensor toType(const Type & t) const {
    if(type().ID() ==t.ID())
      return *this;
    return t.copy(*this);
  }
  Tensor & copy_(const Tensor & src) {
    resize_(src.sizes());
    type().copy(src,*this);
    return *this;
  }
  Tensor toType(ScalarType t) const {
    return toType(type().toScalarType(t));
  }
  Tensor toBackend(Backend b) const {
    return toType(type().toBackend(b));
  }

  template<typename T>
  T * data() const;

  void * unsafeGetTH(bool retain) const {
    return pImpl->unsafeGetTH(retain);
  }

  //toLongData(), toFloatData() etc.
  #define TO_TYPE_DATA(T,name,_) \
  T * to##name##Data() const;
  AT_FORALL_SCALAR_TYPES(TO_TYPE_DATA)
  #undef TO_TYPE_DATA

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

  //example
  //Tensor * add(Tensor & b);
  int64_t storage_offset() const;
  Tensor & resize_(IntList size);
  int64_t numel() const;
  Tensor & set_(Storage & storage);
  Tensor & set_(Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride);
  Tensor & set_(Storage & sourceStorage, int64_t storage_offset, IntList size);
  Tensor & set_(const Tensor & source);
  Tensor & set_();
  Tensor & fill_(Scalar value);
  bool is_same_size(const Tensor & other) const;
  bool is_contiguous() const;
  bool is_set_to(const Tensor & tensor) const;
  Tensor & masked_fill_(const Tensor & mask, Scalar value);
  Tensor & masked_scatter_(const Tensor & mask, const Tensor & source);
  Tensor masked_select(const Tensor & mask) const;
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  Tensor & transpose_(int64_t dim0, int64_t dim1);
  Tensor t() const;
  Tensor & t_();
  Tensor squeeze(int64_t dim) const;
  Tensor squeeze() const;
  Tensor & squeeze_(int64_t dim);
  Tensor & squeeze_();
  Tensor unsqueeze(int64_t dim) const;
  Tensor & unsqueeze_(int64_t dim);
  Tensor nonzero() const;
  Tensor contiguous() const;
  Tensor clone() const;
  Tensor view(IntList size) const;
  Tensor expand(IntList size) const;
  Tensor & resize_as_(const Tensor & the_template);
  Tensor index_select(int64_t dim, const Tensor & index) const;
  Tensor & index_copy_(int64_t dim, const Tensor & index, const Tensor & source);
  Tensor & index_add_(int64_t dim, const Tensor & index, const Tensor & source);
  Tensor & index_fill_(int64_t dim, const Tensor & index, Scalar value);
  Tensor narrow(int64_t dimension, int64_t start, int64_t length) const;
  Tensor unfold(int64_t dimension, int64_t size, int64_t step) const;
  Tensor & scatter_(int64_t dim, const Tensor & index, const Tensor & src);
  Tensor & scatter_(int64_t dim, const Tensor & index, Scalar value);
  Tensor & scatter_add_(int64_t dim, const Tensor & index, const Tensor & src);
  Tensor gather(int64_t dim, const Tensor & index) const;
  void* data_ptr() const;
  bool equal(const Tensor & other) const;
  Tensor __and__(Scalar value) const;
  Tensor __and__(const Tensor & other) const;
  Tensor & __iand__(Scalar value);
  Tensor & __iand__(const Tensor & other);
  Tensor __or__(Scalar value) const;
  Tensor __or__(const Tensor & other) const;
  Tensor & __ior__(Scalar value);
  Tensor & __ior__(const Tensor & other);
  Tensor __xor__(Scalar value) const;
  Tensor __xor__(const Tensor & other) const;
  Tensor & __ixor__(Scalar value);
  Tensor & __ixor__(const Tensor & other);
  Tensor __lshift__(Scalar value) const;
  Tensor __lshift__(const Tensor & other) const;
  Tensor & __ilshift__(Scalar value);
  Tensor & __ilshift__(const Tensor & other);
  Tensor __rshift__(Scalar value) const;
  Tensor __rshift__(const Tensor & other) const;
  Tensor & __irshift__(Scalar value);
  Tensor & __irshift__(const Tensor & other);
  Tensor lt(Scalar value) const;
  Tensor lt(const Tensor & other) const;
  Tensor & lt_(Scalar value);
  Tensor & lt_(const Tensor & other);
  Tensor gt(Scalar value) const;
  Tensor gt(const Tensor & other) const;
  Tensor & gt_(Scalar value);
  Tensor & gt_(const Tensor & other);
  Tensor le(Scalar value) const;
  Tensor le(const Tensor & other) const;
  Tensor & le_(Scalar value);
  Tensor & le_(const Tensor & other);
  Tensor ge(Scalar value) const;
  Tensor ge(const Tensor & other) const;
  Tensor & ge_(Scalar value);
  Tensor & ge_(const Tensor & other);
  Tensor eq(Scalar value) const;
  Tensor eq(const Tensor & other) const;
  Tensor & eq_(Scalar value);
  Tensor & eq_(const Tensor & other);
  Tensor ne(Scalar value) const;
  Tensor ne(const Tensor & other) const;
  Tensor & ne_(Scalar value);
  Tensor & ne_(const Tensor & other);
  std::tuple<Tensor,Tensor> min(int64_t dim, bool keepdim) const;
  std::tuple<Tensor,Tensor> min(int64_t dim) const;
  Tensor min(const Tensor & other) const;
  Scalar min() const;
  std::tuple<Tensor,Tensor> max(int64_t dim, bool keepdim) const;
  std::tuple<Tensor,Tensor> max(int64_t dim) const;
  Tensor max(const Tensor & other) const;
  Scalar max() const;
  std::tuple<Tensor,Tensor> kthvalue(int64_t k, bool keepdim) const;
  std::tuple<Tensor,Tensor> kthvalue(int64_t k) const;
  std::tuple<Tensor,Tensor> kthvalue(int64_t k, int64_t dim, bool keepdim) const;
  std::tuple<Tensor,Tensor> kthvalue(int64_t k, int64_t dim) const;
  std::tuple<Tensor,Tensor> mode(bool keepdim) const;
  std::tuple<Tensor,Tensor> mode() const;
  std::tuple<Tensor,Tensor> mode(int64_t dim, bool keepdim) const;
  std::tuple<Tensor,Tensor> mode(int64_t dim) const;
  std::tuple<Tensor,Tensor> median(bool keepdim) const;
  std::tuple<Tensor,Tensor> median(int64_t dim) const;
  std::tuple<Tensor,Tensor> median(int64_t dim, bool keepdim) const;
  Scalar median() const;
  std::tuple<Tensor,Tensor> sort() const;
  std::tuple<Tensor,Tensor> sort(int64_t dim) const;
  std::tuple<Tensor,Tensor> sort(int64_t dim, bool descending) const;
  std::tuple<Tensor,Tensor> topk(int64_t k) const;
  std::tuple<Tensor,Tensor> topk(int64_t k, int64_t dim, bool largest, bool sorted) const;
  std::tuple<Tensor,Tensor> topk(int64_t k, int64_t dim, bool largest) const;
  std::tuple<Tensor,Tensor> topk(int64_t k, int64_t dim) const;
  bool all() const;
  bool any() const;
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
  Tensor & exp_();
  Tensor exp() const;
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
  Tensor mean(int64_t dim, bool keepdim) const;
  Tensor mean(int64_t dim) const;
  Scalar mean() const;
  Tensor var(int64_t dim, bool unbiased, bool keepdim) const;
  Tensor var(int64_t dim, bool keepdim) const;
  Tensor var(int64_t dim) const;
  Scalar var(bool unbiased) const;
  Scalar var() const;
  Tensor std(int64_t dim, bool unbiased, bool keepdim) const;
  Tensor std(int64_t dim, bool keepdim) const;
  Tensor std(int64_t dim) const;
  Scalar std(bool unbiased) const;
  Scalar std() const;
  Tensor norm(Scalar p, int64_t dim, bool keepdim) const;
  Tensor norm(Scalar p, int64_t dim) const;
  Scalar norm(Scalar p) const;
  Scalar norm() const;
  Tensor renorm(Scalar p, int64_t dim, Scalar maxnorm) const;
  Tensor & renorm_(Scalar p, int64_t dim, Scalar maxnorm);
  Scalar dist(const Tensor & other, Scalar p) const;
  Scalar dist(const Tensor & other) const;
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
  Tensor histc() const;
  Tensor histc(int64_t bins) const;
  Tensor histc(int64_t bins, Scalar min) const;
  Tensor histc(int64_t bins, Scalar min, Scalar max) const;
  Tensor & zero_();
  Tensor sum(int64_t dim, bool keepdim) const;
  Tensor sum(int64_t dim) const;
  Scalar sum() const;
  Tensor prod(int64_t dim, bool keepdim) const;
  Tensor prod(int64_t dim) const;
  Scalar prod() const;
  Tensor cumsum(int64_t dim) const;
  Tensor cumprod(int64_t dim) const;
  Tensor sign() const;
  Tensor & sign_();
  Scalar trace() const;
  Tensor add(Scalar value, const Tensor & other) const;
  Tensor add(Scalar value, SparseTensor other) const;
  Tensor add(Scalar value) const;
  Tensor add(const Tensor & other) const;
  Tensor add(SparseTensor other) const;
  Tensor & add_(Scalar value, const Tensor & other);
  Tensor & add_(Scalar value, SparseTensor other);
  Tensor & add_(Scalar value);
  Tensor & add_(const Tensor & other);
  Tensor & add_(SparseTensor other);
  Tensor sub(Scalar value, const Tensor & other) const;
  Tensor sub(Scalar value) const;
  Tensor sub(const Tensor & other) const;
  Tensor & sub_(Scalar value, const Tensor & other);
  Tensor & sub_(Scalar value);
  Tensor & sub_(const Tensor & other);
  Tensor mul(Scalar value) const;
  Tensor mul(const Tensor & other) const;
  Tensor & mul_(Scalar value);
  Tensor & mul_(const Tensor & other);
  Tensor div(Scalar value) const;
  Tensor div(const Tensor & other) const;
  Tensor & div_(Scalar value);
  Tensor & div_(const Tensor & other);
  Tensor fmod(Scalar value) const;
  Tensor fmod(const Tensor & other) const;
  Tensor & fmod_(Scalar value);
  Tensor & fmod_(const Tensor & other);
  Tensor remainder(Scalar value) const;
  Tensor remainder(const Tensor & other) const;
  Tensor & remainder_(Scalar value);
  Tensor & remainder_(const Tensor & other);
  Tensor clamp(Scalar min, Scalar max) const;
  Tensor clamp(Scalar min) const;
  Tensor & clamp_(Scalar min, Scalar max);
  Tensor & clamp_(Scalar min);
  Scalar dot(const Tensor & tensor) const;
  Tensor tril(int64_t diagonal) const;
  Tensor tril() const;
  Tensor & tril_(int64_t diagonal);
  Tensor & tril_();
  Tensor triu(int64_t diagonal) const;
  Tensor triu() const;
  Tensor & triu_(int64_t diagonal);
  Tensor & triu_();
  Tensor cross(const Tensor & other, int64_t dim) const;
  Tensor cross(const Tensor & other) const;
  Tensor diag(int64_t diagonal) const;
  Tensor diag() const;
  Tensor addmm(Scalar beta, Scalar alpha, const Tensor & mat1, const Tensor & mat2) const;
  Tensor addmm(Scalar beta, const Tensor & mat1, const Tensor & mat2) const;
  Tensor addmm(const Tensor & mat1, const Tensor & mat2) const;
  Tensor & addmm_(Scalar beta, Scalar alpha, const Tensor & mat1, const Tensor & mat2);
  Tensor & addmm_(Scalar beta, const Tensor & mat1, const Tensor & mat2);
  Tensor & addmm_(const Tensor & mat1, const Tensor & mat2);
  Tensor addmv(Scalar beta, Scalar alpha, const Tensor & mat, const Tensor & vec) const;
  Tensor addmv(Scalar beta, const Tensor & mat, const Tensor & vec) const;
  Tensor addmv(const Tensor & mat, const Tensor & vec) const;
  Tensor & addmv_(Scalar beta, Scalar alpha, const Tensor & mat, const Tensor & vec);
  Tensor & addmv_(Scalar beta, const Tensor & mat, const Tensor & vec);
  Tensor & addmv_(const Tensor & mat, const Tensor & vec);
  Tensor addr(Scalar beta, Scalar alpha, const Tensor & vec1, const Tensor & vec2) const;
  Tensor addr(Scalar beta, const Tensor & vec1, const Tensor & vec2) const;
  Tensor addr(const Tensor & vec1, const Tensor & vec2) const;
  Tensor & addr_(Scalar beta, Scalar alpha, const Tensor & vec1, const Tensor & vec2);
  Tensor & addr_(Scalar beta, const Tensor & vec1, const Tensor & vec2);
  Tensor & addr_(const Tensor & vec1, const Tensor & vec2);
  Tensor ger(const Tensor & vec2) const;
  Tensor mv(const Tensor & vec) const;
  Tensor mm(const Tensor & mat2) const;
  Tensor bmm(const Tensor & mat2) const;
  Tensor addbmm(Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) const;
  Tensor addbmm(Scalar beta, const Tensor & batch1, const Tensor & batch2) const;
  Tensor addbmm(const Tensor & batch1, const Tensor & batch2) const;
  Tensor & addbmm_(Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2);
  Tensor & addbmm_(Scalar beta, const Tensor & batch1, const Tensor & batch2);
  Tensor & addbmm_(const Tensor & batch1, const Tensor & batch2);
  Tensor baddbmm(Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) const;
  Tensor baddbmm(Scalar beta, const Tensor & batch1, const Tensor & batch2) const;
  Tensor baddbmm(const Tensor & batch1, const Tensor & batch2) const;
  Tensor & baddbmm_(Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2);
  Tensor & baddbmm_(Scalar beta, const Tensor & batch1, const Tensor & batch2);
  Tensor & baddbmm_(const Tensor & batch1, const Tensor & batch2);
  Tensor addcmul(Scalar value, const Tensor & tensor1, const Tensor & tensor2) const;
  Tensor addcmul(const Tensor & tensor1, const Tensor & tensor2) const;
  Tensor & addcmul_(Scalar value, const Tensor & tensor1, const Tensor & tensor2);
  Tensor & addcmul_(const Tensor & tensor1, const Tensor & tensor2);
  Tensor addcdiv(Scalar value, const Tensor & tensor1, const Tensor & tensor2) const;
  Tensor addcdiv(const Tensor & tensor1, const Tensor & tensor2) const;
  Tensor & addcdiv_(Scalar value, const Tensor & tensor1, const Tensor & tensor2);
  Tensor & addcdiv_(const Tensor & tensor1, const Tensor & tensor2);
  std::tuple<Tensor,Tensor> gesv(const Tensor & A) const;
  std::tuple<Tensor,Tensor> gels(const Tensor & A) const;
  std::tuple<Tensor,Tensor> trtrs(const Tensor & A, bool upper, bool transpose, bool unitriangular) const;
  std::tuple<Tensor,Tensor> trtrs(const Tensor & A, bool upper, bool transpose) const;
  std::tuple<Tensor,Tensor> trtrs(const Tensor & A, bool upper) const;
  std::tuple<Tensor,Tensor> trtrs(const Tensor & A) const;
  std::tuple<Tensor,Tensor> symeig(bool eigenvectors, bool upper) const;
  std::tuple<Tensor,Tensor> symeig(bool eigenvectors) const;
  std::tuple<Tensor,Tensor> symeig() const;
  std::tuple<Tensor,Tensor> eig(bool eigenvectors) const;
  std::tuple<Tensor,Tensor> eig() const;
  std::tuple<Tensor,Tensor,Tensor> svd(bool some) const;
  std::tuple<Tensor,Tensor,Tensor> svd() const;
  Tensor inverse() const;
  Tensor potrf(bool upper) const;
  Tensor potrf() const;
  Tensor potrs(const Tensor & input2, bool upper) const;
  Tensor potrs(const Tensor & input2) const;
  Tensor potri(bool upper) const;
  Tensor potri() const;
  std::tuple<Tensor,Tensor> pstrf(bool upper, Scalar tol) const;
  std::tuple<Tensor,Tensor> pstrf(bool upper) const;
  std::tuple<Tensor,Tensor> pstrf(Scalar tol) const;
  std::tuple<Tensor,Tensor> pstrf() const;
  std::tuple<Tensor,Tensor> qr() const;
  std::tuple<Tensor,Tensor> geqrf() const;
  std::tuple<Tensor,const Tensor &> orgqr(const Tensor & input2) const;
  std::tuple<Tensor,const Tensor &> ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const;
  std::tuple<Tensor,const Tensor &> ormqr(const Tensor & input2, const Tensor & input3, bool left) const;
  std::tuple<Tensor,const Tensor &> ormqr(const Tensor & input2, const Tensor & input3) const;
  std::tuple<Tensor,Tensor> btrifact(const Tensor & info, bool pivot) const;
  std::tuple<Tensor,Tensor> btrifact(const Tensor & info) const;
  std::tuple<Tensor,Tensor> btrifact(bool pivot) const;
  std::tuple<Tensor,Tensor> btrifact() const;
  Tensor btrisolve(const Tensor & LU_data, const Tensor & LU_pivots) const;
  Tensor multinomial(Generator & generator, int64_t num_samples, bool replacement) const;
  Tensor multinomial(Generator & generator, int64_t num_samples) const;
  Tensor multinomial(int64_t num_samples, bool replacement) const;
  Tensor multinomial(int64_t num_samples) const;
  Tensor & uniform_(Generator & generator, double from, double to);
  Tensor & uniform_(Generator & generator, double from);
  Tensor & uniform_(double from, double to);
  Tensor & uniform_(Generator & generator);
  Tensor & uniform_(double from);
  Tensor & uniform_();
  Tensor & cauchy_(Generator & generator, double median, double sigma);
  Tensor & cauchy_(Generator & generator, double median);
  Tensor & cauchy_(double median, double sigma);
  Tensor & cauchy_(Generator & generator);
  Tensor & cauchy_(double median);
  Tensor & cauchy_();
  Tensor & log_normal_(Generator & generator, double mean, double std);
  Tensor & log_normal_(Generator & generator, double mean);
  Tensor & log_normal_(double mean, double std);
  Tensor & log_normal_(Generator & generator);
  Tensor & log_normal_(double mean);
  Tensor & log_normal_();
  Tensor & geometric_(Generator & generator, double p);
  Tensor & geometric_(double p);
  int64_t size(int64_t dim) const;
  int64_t stride(int64_t dim) const;
  Tensor select(int dim, int64_t sliceIndex) const;
  Tensor & assign_(const Tensor & src);

  friend struct Type;

//TODO(zach): sort out friend structes
public:
  TensorImpl * pImpl;
};

} //namespace at

#pragma once

#include "ATen/Scalar.h"
#include "ATen/Type.h"
#include "ATen/TensorImpl.h"
#include "ATen/Utils.h"
#include "ATen/TensorAccessor.h"

namespace at {
class Type;

struct Tensor {

  Tensor()
  : pImpl(nullptr){}
  explicit Tensor(TensorImpl * self, bool retain = true)
  : pImpl(self) {
    if(pImpl != nullptr && retain)
      pImpl->retain();
  }
  Tensor(Tensor const & rhs)
  : pImpl(rhs.pImpl) {
    if(pImpl != nullptr)
      pImpl->retain();
  }
  Tensor(Tensor && rhs)
  : pImpl(rhs.pImpl) {
    rhs.pImpl = nullptr;
  }
  ~Tensor() {
    if(pImpl != nullptr)
      pImpl->release();
  }
  Tensor & operator=(Tensor && rhs) {
    rhs.swap(*this);
    return *this;
  }
  Tensor & operator=(Tensor const & rhs) {
      //Tensor ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally Tensor dtor releases rhs.pImpl, which was originally this->pImpl
      Tensor(rhs).swap(*this);
      return *this;
  }
  void reset() {
    Tensor().swap(*this);
  }
  void reset(TensorImpl * rhs) {
    Tensor(rhs).swap(*this);
  }
  void reset(TensorImpl * rhs, bool retain) {
    Tensor(rhs, retain).swap(*this );
  }
  TensorImpl * get() {
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
  Type & type() const {
    return pImpl->type();
  }
  Tensor toType(Type & t) const {
    if(type().ID() ==t.ID())
      return *this;
    return t.copy(*this);
  }
  Tensor & copy_(const Tensor & src) {
    resize_(src.sizes());
    type().copy(src,*this);
    return *this;
  }
  Tensor toType(ScalarType t) {
    return toType(type().toScalarType(t));
  }
  Tensor toBackend(Backend b) {
    return toType(type().toBackend(b));
  }
  int64_t dim() const {
    return ndimension();
  }
  template<typename T>
  T * data() const;

  template<typename T, size_t N>
  TensorAccessor<T,N> accessor() {
    static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data<T>()");
    AT_ASSERT(dim() == N, "expected %d dims but tensor has %d",N,dim());
    return TensorAccessor<T,N>(data<T>(),sizes().data(),strides().data());
  }

  Tensor operator-() {
    return neg();
  }
  Tensor& operator+=(const Tensor & other) {
    add_(other);
  }
  Tensor& operator+=(Scalar other) {
    add_(other);
  }
  Tensor& operator-=(const Tensor & other) {
    sub_(other);
  }
  Tensor& operator-=(Scalar other) {
    sub_(other);
  }
  Tensor& operator*=(const Tensor & other) {
    mul_(other);
  }
  Tensor& operator*=(Scalar other) {
    mul_(other);
  }
  Tensor& operator/=(const Tensor & other) {
    div_(other);
  }
  Tensor& operator/=(Scalar other) {
    div_(other);
  }
  Tensor operator[](int64_t idx) {
    return select(0,idx);
  }

  //example
  //Tensor * add(Tensor & b);
  int64_t storage_offset() const;
  int64_t ndimension() const;
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
  std::tuple<Tensor,Tensor> median() const;
  std::tuple<Tensor,Tensor> median(int64_t dim) const;
  std::tuple<Tensor,Tensor> median(int64_t dim, bool keepdim) const;
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
  Tensor var(int64_t dim, bool keepdim) const;
  Tensor var(int64_t dim) const;
  Scalar var() const;
  Tensor std(int64_t dim, bool keepdim) const;
  Tensor std(int64_t dim) const;
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
  Tensor add(Scalar value) const;
  Tensor add(const Tensor & other) const;
  Tensor & add_(Scalar value, const Tensor & other);
  Tensor & add_(Scalar value);
  Tensor & add_(const Tensor & other);
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
  Tensor tril(int64_t k) const;
  Tensor tril() const;
  Tensor & tril_(int64_t k);
  Tensor & tril_();
  Tensor triu(int64_t k) const;
  Tensor triu() const;
  Tensor & triu_(int64_t k);
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
  std::tuple<Tensor,Tensor> btrifact(const Tensor & info) const;
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

  friend class Type;

//TODO(zach): sort out friend classes
public:
  TensorImpl * pImpl;
};

// all static inline to allow for inlining of the non-dynamic part of dispatch
inline int64_t Tensor::storage_offset() const {
    return type().m_storage_offset(*this);
}
inline int64_t Tensor::ndimension() const {
    return type().m_ndimension(*this);
}
inline Tensor & Tensor::resize_(IntList size) {
    return type().m_resize_(*this, size);
}
inline int64_t Tensor::numel() const {
    return type().numel(*this);
}
inline Tensor & Tensor::set_(Storage & storage) {
    return type().m_set_(*this, storage);
}
inline Tensor & Tensor::set_(Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) {
    return type().m_set_(*this, sourceStorage, storage_offset, size, stride);
}
inline Tensor & Tensor::set_(Storage & sourceStorage, int64_t storage_offset, IntList size) {
    return type().m_set_(*this, sourceStorage, storage_offset, size);
}
inline Tensor & Tensor::set_(const Tensor & source) {
    return type().m_set_(*this, source);
}
inline Tensor & Tensor::set_() {
    return type().m_set_(*this);
}
inline Tensor & Tensor::fill_(Scalar value) {
    return type().m_fill_(*this, value);
}
inline bool Tensor::is_same_size(const Tensor & other) const {
    return type().m_is_same_size(*this, other);
}
inline bool Tensor::is_contiguous() const {
    return type().m_is_contiguous(*this);
}
inline bool Tensor::is_set_to(const Tensor & tensor) const {
    return type().m_is_set_to(*this, tensor);
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, Scalar value) {
    return type().m_masked_fill_(*this, mask, value);
}
inline Tensor & Tensor::masked_scatter_(const Tensor & mask, const Tensor & source) {
    return type().m_masked_scatter_(*this, mask, source);
}
inline Tensor Tensor::masked_select(const Tensor & mask) const {
    return type().masked_select(*this, mask);
}
inline Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    return type().transpose(*this, dim0, dim1);
}
inline Tensor & Tensor::transpose_(int64_t dim0, int64_t dim1) {
    return type().m_transpose_(*this, dim0, dim1);
}
inline Tensor Tensor::t() const {
    return type().t(*this);
}
inline Tensor & Tensor::t_() {
    return type().m_t_(*this);
}
inline Tensor Tensor::squeeze(int64_t dim) const {
    return type().squeeze(*this, dim);
}
inline Tensor Tensor::squeeze() const {
    return type().squeeze(*this);
}
inline Tensor & Tensor::squeeze_(int64_t dim) {
    return type().m_squeeze_(*this, dim);
}
inline Tensor & Tensor::squeeze_() {
    return type().m_squeeze_(*this);
}
inline Tensor Tensor::unsqueeze(int64_t dim) const {
    return type().unsqueeze(*this, dim);
}
inline Tensor & Tensor::unsqueeze_(int64_t dim) {
    return type().m_unsqueeze_(*this, dim);
}
inline Tensor Tensor::nonzero() const {
    return type().nonzero(*this);
}
inline Tensor Tensor::contiguous() const {
    return type().m_contiguous(*this);
}
inline Tensor Tensor::clone() const {
    return type().m_clone(*this);
}
inline Tensor Tensor::view(IntList size) const {
    return type().m_view(*this, size);
}
inline Tensor Tensor::expand(IntList size) const {
    return type().m_expand(*this, size);
}
inline Tensor & Tensor::resize_as_(const Tensor & the_template) {
    return type().m_resize_as_(*this, the_template);
}
inline Tensor Tensor::index_select(int64_t dim, const Tensor & index) const {
    return type().index_select(*this, dim, index);
}
inline Tensor & Tensor::index_copy_(int64_t dim, const Tensor & index, const Tensor & source) {
    return type().m_index_copy_(*this, dim, index, source);
}
inline Tensor & Tensor::index_add_(int64_t dim, const Tensor & index, const Tensor & source) {
    return type().m_index_add_(*this, dim, index, source);
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, Scalar value) {
    return type().m_index_fill_(*this, dim, index, value);
}
inline Tensor Tensor::narrow(int64_t dimension, int64_t start, int64_t length) const {
    return type().m_narrow(*this, dimension, start, length);
}
inline Tensor Tensor::unfold(int64_t dimension, int64_t size, int64_t step) const {
    return type().m_unfold(*this, dimension, size, step);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, const Tensor & src) {
    return type().m_scatter_(*this, dim, index, src);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, Scalar value) {
    return type().m_scatter_(*this, dim, index, value);
}
inline Tensor & Tensor::scatter_add_(int64_t dim, const Tensor & index, const Tensor & src) {
    return type().m_scatter_add_(*this, dim, index, src);
}
inline Tensor Tensor::gather(int64_t dim, const Tensor & index) const {
    return type().gather(*this, dim, index);
}
inline void* Tensor::data_ptr() const {
    return type().m_data_ptr(*this);
}
inline bool Tensor::equal(const Tensor & other) const {
    return type().equal(*this, other);
}
inline Tensor Tensor::__and__(Scalar value) const {
    return type().__and__(*this, value);
}
inline Tensor Tensor::__and__(const Tensor & other) const {
    return type().__and__(*this, other);
}
inline Tensor & Tensor::__iand__(Scalar value) {
    return type().__iand__(*this, value);
}
inline Tensor & Tensor::__iand__(const Tensor & other) {
    return type().__iand__(*this, other);
}
inline Tensor Tensor::__or__(Scalar value) const {
    return type().__or__(*this, value);
}
inline Tensor Tensor::__or__(const Tensor & other) const {
    return type().__or__(*this, other);
}
inline Tensor & Tensor::__ior__(Scalar value) {
    return type().__ior__(*this, value);
}
inline Tensor & Tensor::__ior__(const Tensor & other) {
    return type().__ior__(*this, other);
}
inline Tensor Tensor::__xor__(Scalar value) const {
    return type().__xor__(*this, value);
}
inline Tensor Tensor::__xor__(const Tensor & other) const {
    return type().__xor__(*this, other);
}
inline Tensor & Tensor::__ixor__(Scalar value) {
    return type().__ixor__(*this, value);
}
inline Tensor & Tensor::__ixor__(const Tensor & other) {
    return type().__ixor__(*this, other);
}
inline Tensor Tensor::__lshift__(Scalar value) const {
    return type().__lshift__(*this, value);
}
inline Tensor Tensor::__lshift__(const Tensor & other) const {
    return type().__lshift__(*this, other);
}
inline Tensor & Tensor::__ilshift__(Scalar value) {
    return type().__ilshift__(*this, value);
}
inline Tensor & Tensor::__ilshift__(const Tensor & other) {
    return type().__ilshift__(*this, other);
}
inline Tensor Tensor::__rshift__(Scalar value) const {
    return type().__rshift__(*this, value);
}
inline Tensor Tensor::__rshift__(const Tensor & other) const {
    return type().__rshift__(*this, other);
}
inline Tensor & Tensor::__irshift__(Scalar value) {
    return type().__irshift__(*this, value);
}
inline Tensor & Tensor::__irshift__(const Tensor & other) {
    return type().__irshift__(*this, other);
}
inline Tensor Tensor::lt(Scalar value) const {
    return type().m_lt(*this, value);
}
inline Tensor Tensor::lt(const Tensor & other) const {
    return type().m_lt(*this, other);
}
inline Tensor & Tensor::lt_(Scalar value) {
    return type().m_lt_(*this, value);
}
inline Tensor & Tensor::lt_(const Tensor & other) {
    return type().m_lt_(*this, other);
}
inline Tensor Tensor::gt(Scalar value) const {
    return type().m_gt(*this, value);
}
inline Tensor Tensor::gt(const Tensor & other) const {
    return type().m_gt(*this, other);
}
inline Tensor & Tensor::gt_(Scalar value) {
    return type().m_gt_(*this, value);
}
inline Tensor & Tensor::gt_(const Tensor & other) {
    return type().m_gt_(*this, other);
}
inline Tensor Tensor::le(Scalar value) const {
    return type().m_le(*this, value);
}
inline Tensor Tensor::le(const Tensor & other) const {
    return type().m_le(*this, other);
}
inline Tensor & Tensor::le_(Scalar value) {
    return type().m_le_(*this, value);
}
inline Tensor & Tensor::le_(const Tensor & other) {
    return type().m_le_(*this, other);
}
inline Tensor Tensor::ge(Scalar value) const {
    return type().m_ge(*this, value);
}
inline Tensor Tensor::ge(const Tensor & other) const {
    return type().m_ge(*this, other);
}
inline Tensor & Tensor::ge_(Scalar value) {
    return type().m_ge_(*this, value);
}
inline Tensor & Tensor::ge_(const Tensor & other) {
    return type().m_ge_(*this, other);
}
inline Tensor Tensor::eq(Scalar value) const {
    return type().m_eq(*this, value);
}
inline Tensor Tensor::eq(const Tensor & other) const {
    return type().m_eq(*this, other);
}
inline Tensor & Tensor::eq_(Scalar value) {
    return type().m_eq_(*this, value);
}
inline Tensor & Tensor::eq_(const Tensor & other) {
    return type().m_eq_(*this, other);
}
inline Tensor Tensor::ne(Scalar value) const {
    return type().m_ne(*this, value);
}
inline Tensor Tensor::ne(const Tensor & other) const {
    return type().m_ne(*this, other);
}
inline Tensor & Tensor::ne_(Scalar value) {
    return type().m_ne_(*this, value);
}
inline Tensor & Tensor::ne_(const Tensor & other) {
    return type().m_ne_(*this, other);
}
inline std::tuple<Tensor,Tensor> Tensor::min(int64_t dim, bool keepdim) const {
    return type().min(*this, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::min(int64_t dim) const {
    return type().min(*this, dim);
}
inline Tensor Tensor::min(const Tensor & other) const {
    return type().min(*this, other);
}
inline Scalar Tensor::min() const {
    return type().min(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::max(int64_t dim, bool keepdim) const {
    return type().max(*this, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::max(int64_t dim) const {
    return type().max(*this, dim);
}
inline Tensor Tensor::max(const Tensor & other) const {
    return type().max(*this, other);
}
inline Scalar Tensor::max() const {
    return type().max(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, bool keepdim) const {
    return type().kthvalue(*this, k, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k) const {
    return type().kthvalue(*this, k);
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, int64_t dim, bool keepdim) const {
    return type().kthvalue(*this, k, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, int64_t dim) const {
    return type().kthvalue(*this, k, dim);
}
inline std::tuple<Tensor,Tensor> Tensor::mode(bool keepdim) const {
    return type().mode(*this, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::mode() const {
    return type().mode(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::mode(int64_t dim, bool keepdim) const {
    return type().mode(*this, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::mode(int64_t dim) const {
    return type().mode(*this, dim);
}
inline std::tuple<Tensor,Tensor> Tensor::median(bool keepdim) const {
    return type().median(*this, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::median() const {
    return type().median(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::median(int64_t dim) const {
    return type().median(*this, dim);
}
inline std::tuple<Tensor,Tensor> Tensor::median(int64_t dim, bool keepdim) const {
    return type().median(*this, dim, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::sort() const {
    return type().sort(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::sort(int64_t dim) const {
    return type().sort(*this, dim);
}
inline std::tuple<Tensor,Tensor> Tensor::sort(int64_t dim, bool descending) const {
    return type().sort(*this, dim, descending);
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k) const {
    return type().topk(*this, k);
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const {
    return type().topk(*this, k, dim, largest, sorted);
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest) const {
    return type().topk(*this, k, dim, largest);
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim) const {
    return type().topk(*this, k, dim);
}
inline bool Tensor::all() const {
    return type().m_all(*this);
}
inline bool Tensor::any() const {
    return type().m_any(*this);
}
inline int64_t Tensor::get_device() const {
    return type().m_get_device(*this);
}
inline Tensor Tensor::abs() const {
    return type().abs(*this);
}
inline Tensor & Tensor::abs_() {
    return type().m_abs_(*this);
}
inline Tensor & Tensor::sigmoid_() {
    return type().m_sigmoid_(*this);
}
inline Tensor Tensor::sigmoid() const {
    return type().sigmoid(*this);
}
inline Tensor & Tensor::log_() {
    return type().m_log_(*this);
}
inline Tensor Tensor::log() const {
    return type().log(*this);
}
inline Tensor & Tensor::log1p_() {
    return type().m_log1p_(*this);
}
inline Tensor Tensor::log1p() const {
    return type().log1p(*this);
}
inline Tensor Tensor::lgamma() const {
    return type().lgamma(*this);
}
inline Tensor & Tensor::lgamma_() {
    return type().m_lgamma_(*this);
}
inline Tensor & Tensor::exp_() {
    return type().m_exp_(*this);
}
inline Tensor Tensor::exp() const {
    return type().exp(*this);
}
inline Tensor & Tensor::cos_() {
    return type().m_cos_(*this);
}
inline Tensor Tensor::cos() const {
    return type().cos(*this);
}
inline Tensor & Tensor::acos_() {
    return type().m_acos_(*this);
}
inline Tensor Tensor::acos() const {
    return type().acos(*this);
}
inline Tensor & Tensor::cosh_() {
    return type().m_cosh_(*this);
}
inline Tensor Tensor::cosh() const {
    return type().cosh(*this);
}
inline Tensor & Tensor::sin_() {
    return type().m_sin_(*this);
}
inline Tensor Tensor::sin() const {
    return type().sin(*this);
}
inline Tensor & Tensor::asin_() {
    return type().m_asin_(*this);
}
inline Tensor Tensor::asin() const {
    return type().asin(*this);
}
inline Tensor & Tensor::sinh_() {
    return type().m_sinh_(*this);
}
inline Tensor Tensor::sinh() const {
    return type().sinh(*this);
}
inline Tensor & Tensor::tan_() {
    return type().m_tan_(*this);
}
inline Tensor Tensor::tan() const {
    return type().tan(*this);
}
inline Tensor & Tensor::atan_() {
    return type().m_atan_(*this);
}
inline Tensor Tensor::atan() const {
    return type().atan(*this);
}
inline Tensor & Tensor::tanh_() {
    return type().m_tanh_(*this);
}
inline Tensor Tensor::tanh() const {
    return type().tanh(*this);
}
inline Tensor & Tensor::sqrt_() {
    return type().m_sqrt_(*this);
}
inline Tensor Tensor::sqrt() const {
    return type().sqrt(*this);
}
inline Tensor & Tensor::rsqrt_() {
    return type().m_rsqrt_(*this);
}
inline Tensor Tensor::rsqrt() const {
    return type().rsqrt(*this);
}
inline Tensor & Tensor::ceil_() {
    return type().m_ceil_(*this);
}
inline Tensor Tensor::ceil() const {
    return type().ceil(*this);
}
inline Tensor & Tensor::floor_() {
    return type().m_floor_(*this);
}
inline Tensor Tensor::floor() const {
    return type().floor(*this);
}
inline Tensor & Tensor::round_() {
    return type().m_round_(*this);
}
inline Tensor Tensor::round() const {
    return type().round(*this);
}
inline Tensor & Tensor::trunc_() {
    return type().m_trunc_(*this);
}
inline Tensor Tensor::trunc() const {
    return type().trunc(*this);
}
inline Tensor & Tensor::frac_() {
    return type().m_frac_(*this);
}
inline Tensor Tensor::frac() const {
    return type().frac(*this);
}
inline Tensor Tensor::mean(int64_t dim, bool keepdim) const {
    return type().mean(*this, dim, keepdim);
}
inline Tensor Tensor::mean(int64_t dim) const {
    return type().mean(*this, dim);
}
inline Scalar Tensor::mean() const {
    return type().mean(*this);
}
inline Tensor Tensor::var(int64_t dim, bool keepdim) const {
    return type().var(*this, dim, keepdim);
}
inline Tensor Tensor::var(int64_t dim) const {
    return type().var(*this, dim);
}
inline Scalar Tensor::var() const {
    return type().var(*this);
}
inline Tensor Tensor::std(int64_t dim, bool keepdim) const {
    return type().std(*this, dim, keepdim);
}
inline Tensor Tensor::std(int64_t dim) const {
    return type().std(*this, dim);
}
inline Scalar Tensor::std() const {
    return type().std(*this);
}
inline Tensor Tensor::norm(Scalar p, int64_t dim, bool keepdim) const {
    return type().norm(*this, p, dim, keepdim);
}
inline Tensor Tensor::norm(Scalar p, int64_t dim) const {
    return type().norm(*this, p, dim);
}
inline Scalar Tensor::norm(Scalar p) const {
    return type().norm(*this, p);
}
inline Scalar Tensor::norm() const {
    return type().norm(*this);
}
inline Tensor Tensor::renorm(Scalar p, int64_t dim, Scalar maxnorm) const {
    return type().renorm(*this, p, dim, maxnorm);
}
inline Tensor & Tensor::renorm_(Scalar p, int64_t dim, Scalar maxnorm) {
    return type().m_renorm_(*this, p, dim, maxnorm);
}
inline Scalar Tensor::dist(const Tensor & other, Scalar p) const {
    return type().dist(*this, other, p);
}
inline Scalar Tensor::dist(const Tensor & other) const {
    return type().dist(*this, other);
}
inline Tensor Tensor::reciprocal() const {
    return type().reciprocal(*this);
}
inline Tensor & Tensor::reciprocal_() {
    return type().m_reciprocal_(*this);
}
inline Tensor Tensor::neg() const {
    return type().neg(*this);
}
inline Tensor & Tensor::neg_() {
    return type().m_neg_(*this);
}
inline Tensor Tensor::atan2(const Tensor & other) const {
    return type().atan2(*this, other);
}
inline Tensor & Tensor::atan2_(const Tensor & other) {
    return type().m_atan2_(*this, other);
}
inline Tensor Tensor::pow(Scalar exponent) const {
    return type().pow(*this, exponent);
}
inline Tensor Tensor::pow(const Tensor & exponent) const {
    return type().pow(*this, exponent);
}
inline Tensor & Tensor::pow_(Scalar exponent) {
    return type().m_pow_(*this, exponent);
}
inline Tensor & Tensor::pow_(const Tensor & exponent) {
    return type().m_pow_(*this, exponent);
}
inline Tensor Tensor::lerp(const Tensor & end, Scalar weight) const {
    return type().lerp(*this, end, weight);
}
inline Tensor & Tensor::lerp_(const Tensor & end, Scalar weight) {
    return type().m_lerp_(*this, end, weight);
}
inline Tensor Tensor::histc() const {
    return type().histc(*this);
}
inline Tensor Tensor::histc(int64_t bins) const {
    return type().histc(*this, bins);
}
inline Tensor Tensor::histc(int64_t bins, Scalar min) const {
    return type().histc(*this, bins, min);
}
inline Tensor Tensor::histc(int64_t bins, Scalar min, Scalar max) const {
    return type().histc(*this, bins, min, max);
}
inline Tensor & Tensor::zero_() {
    return type().m_zero_(*this);
}
inline Tensor Tensor::sum(int64_t dim, bool keepdim) const {
    return type().sum(*this, dim, keepdim);
}
inline Tensor Tensor::sum(int64_t dim) const {
    return type().sum(*this, dim);
}
inline Scalar Tensor::sum() const {
    return type().sum(*this);
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim) const {
    return type().prod(*this, dim, keepdim);
}
inline Tensor Tensor::prod(int64_t dim) const {
    return type().prod(*this, dim);
}
inline Scalar Tensor::prod() const {
    return type().prod(*this);
}
inline Tensor Tensor::cumsum(int64_t dim) const {
    return type().cumsum(*this, dim);
}
inline Tensor Tensor::cumprod(int64_t dim) const {
    return type().cumprod(*this, dim);
}
inline Tensor Tensor::sign() const {
    return type().sign(*this);
}
inline Tensor & Tensor::sign_() {
    return type().m_sign_(*this);
}
inline Scalar Tensor::trace() const {
    return type().trace(*this);
}
inline Tensor Tensor::add(Scalar value, const Tensor & other) const {
    return type().add(*this, value, other);
}
inline Tensor Tensor::add(Scalar value) const {
    return type().add(*this, value);
}
inline Tensor Tensor::add(const Tensor & other) const {
    return type().add(*this, other);
}
inline Tensor & Tensor::add_(Scalar value, const Tensor & other) {
    return type().m_add_(*this, value, other);
}
inline Tensor & Tensor::add_(Scalar value) {
    return type().m_add_(*this, value);
}
inline Tensor & Tensor::add_(const Tensor & other) {
    return type().m_add_(*this, other);
}
inline Tensor Tensor::sub(Scalar value, const Tensor & other) const {
    return type().sub(*this, value, other);
}
inline Tensor Tensor::sub(Scalar value) const {
    return type().sub(*this, value);
}
inline Tensor Tensor::sub(const Tensor & other) const {
    return type().sub(*this, other);
}
inline Tensor & Tensor::sub_(Scalar value, const Tensor & other) {
    return type().m_sub_(*this, value, other);
}
inline Tensor & Tensor::sub_(Scalar value) {
    return type().m_sub_(*this, value);
}
inline Tensor & Tensor::sub_(const Tensor & other) {
    return type().m_sub_(*this, other);
}
inline Tensor Tensor::mul(Scalar value) const {
    return type().mul(*this, value);
}
inline Tensor Tensor::mul(const Tensor & other) const {
    return type().mul(*this, other);
}
inline Tensor & Tensor::mul_(Scalar value) {
    return type().m_mul_(*this, value);
}
inline Tensor & Tensor::mul_(const Tensor & other) {
    return type().m_mul_(*this, other);
}
inline Tensor Tensor::div(Scalar value) const {
    return type().div(*this, value);
}
inline Tensor Tensor::div(const Tensor & other) const {
    return type().div(*this, other);
}
inline Tensor & Tensor::div_(Scalar value) {
    return type().m_div_(*this, value);
}
inline Tensor & Tensor::div_(const Tensor & other) {
    return type().m_div_(*this, other);
}
inline Tensor Tensor::fmod(Scalar value) const {
    return type().fmod(*this, value);
}
inline Tensor Tensor::fmod(const Tensor & other) const {
    return type().fmod(*this, other);
}
inline Tensor & Tensor::fmod_(Scalar value) {
    return type().m_fmod_(*this, value);
}
inline Tensor & Tensor::fmod_(const Tensor & other) {
    return type().m_fmod_(*this, other);
}
inline Tensor Tensor::remainder(Scalar value) const {
    return type().remainder(*this, value);
}
inline Tensor Tensor::remainder(const Tensor & other) const {
    return type().remainder(*this, other);
}
inline Tensor & Tensor::remainder_(Scalar value) {
    return type().m_remainder_(*this, value);
}
inline Tensor & Tensor::remainder_(const Tensor & other) {
    return type().m_remainder_(*this, other);
}
inline Tensor Tensor::clamp(Scalar min, Scalar max) const {
    return type().clamp(*this, min, max);
}
inline Tensor Tensor::clamp(Scalar min) const {
    return type().clamp(*this, min);
}
inline Tensor & Tensor::clamp_(Scalar min, Scalar max) {
    return type().m_clamp_(*this, min, max);
}
inline Tensor & Tensor::clamp_(Scalar min) {
    return type().m_clamp_(*this, min);
}
inline Scalar Tensor::dot(const Tensor & tensor) const {
    return type().dot(*this, tensor);
}
inline Tensor Tensor::tril(int64_t k) const {
    return type().tril(*this, k);
}
inline Tensor Tensor::tril() const {
    return type().tril(*this);
}
inline Tensor & Tensor::tril_(int64_t k) {
    return type().m_tril_(*this, k);
}
inline Tensor & Tensor::tril_() {
    return type().m_tril_(*this);
}
inline Tensor Tensor::triu(int64_t k) const {
    return type().triu(*this, k);
}
inline Tensor Tensor::triu() const {
    return type().triu(*this);
}
inline Tensor & Tensor::triu_(int64_t k) {
    return type().m_triu_(*this, k);
}
inline Tensor & Tensor::triu_() {
    return type().m_triu_(*this);
}
inline Tensor Tensor::cross(const Tensor & other, int64_t dim) const {
    return type().cross(*this, other, dim);
}
inline Tensor Tensor::cross(const Tensor & other) const {
    return type().cross(*this, other);
}
inline Tensor Tensor::diag(int64_t diagonal) const {
    return type().diag(*this, diagonal);
}
inline Tensor Tensor::diag() const {
    return type().diag(*this);
}
inline Tensor Tensor::addmm(Scalar beta, Scalar alpha, const Tensor & mat1, const Tensor & mat2) const {
    return type().addmm(beta, *this, alpha, mat1, mat2);
}
inline Tensor Tensor::addmm(Scalar beta, const Tensor & mat1, const Tensor & mat2) const {
    return type().addmm(beta, *this, mat1, mat2);
}
inline Tensor Tensor::addmm(const Tensor & mat1, const Tensor & mat2) const {
    return type().addmm(*this, mat1, mat2);
}
inline Tensor & Tensor::addmm_(Scalar beta, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {
    return type().m_addmm_(*this, beta, alpha, mat1, mat2);
}
inline Tensor & Tensor::addmm_(Scalar beta, const Tensor & mat1, const Tensor & mat2) {
    return type().m_addmm_(*this, beta, mat1, mat2);
}
inline Tensor & Tensor::addmm_(const Tensor & mat1, const Tensor & mat2) {
    return type().m_addmm_(*this, mat1, mat2);
}
inline Tensor Tensor::addmv(Scalar beta, Scalar alpha, const Tensor & mat, const Tensor & vec) const {
    return type().addmv(beta, *this, alpha, mat, vec);
}
inline Tensor Tensor::addmv(Scalar beta, const Tensor & mat, const Tensor & vec) const {
    return type().addmv(beta, *this, mat, vec);
}
inline Tensor Tensor::addmv(const Tensor & mat, const Tensor & vec) const {
    return type().addmv(*this, mat, vec);
}
inline Tensor & Tensor::addmv_(Scalar beta, Scalar alpha, const Tensor & mat, const Tensor & vec) {
    return type().m_addmv_(*this, beta, alpha, mat, vec);
}
inline Tensor & Tensor::addmv_(Scalar beta, const Tensor & mat, const Tensor & vec) {
    return type().m_addmv_(*this, beta, mat, vec);
}
inline Tensor & Tensor::addmv_(const Tensor & mat, const Tensor & vec) {
    return type().m_addmv_(*this, mat, vec);
}
inline Tensor Tensor::addr(Scalar beta, Scalar alpha, const Tensor & vec1, const Tensor & vec2) const {
    return type().addr(beta, *this, alpha, vec1, vec2);
}
inline Tensor Tensor::addr(Scalar beta, const Tensor & vec1, const Tensor & vec2) const {
    return type().addr(beta, *this, vec1, vec2);
}
inline Tensor Tensor::addr(const Tensor & vec1, const Tensor & vec2) const {
    return type().addr(*this, vec1, vec2);
}
inline Tensor & Tensor::addr_(Scalar beta, Scalar alpha, const Tensor & vec1, const Tensor & vec2) {
    return type().m_addr_(*this, beta, alpha, vec1, vec2);
}
inline Tensor & Tensor::addr_(Scalar beta, const Tensor & vec1, const Tensor & vec2) {
    return type().m_addr_(*this, beta, vec1, vec2);
}
inline Tensor & Tensor::addr_(const Tensor & vec1, const Tensor & vec2) {
    return type().m_addr_(*this, vec1, vec2);
}
inline Tensor Tensor::ger(const Tensor & vec2) const {
    return type().ger(*this, vec2);
}
inline Tensor Tensor::mv(const Tensor & vec) const {
    return type().mv(*this, vec);
}
inline Tensor Tensor::mm(const Tensor & mat2) const {
    return type().mm(*this, mat2);
}
inline Tensor Tensor::bmm(const Tensor & mat2) const {
    return type().bmm(*this, mat2);
}
inline Tensor Tensor::addbmm(Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) const {
    return type().addbmm(beta, *this, alpha, batch1, batch2);
}
inline Tensor Tensor::addbmm(Scalar beta, const Tensor & batch1, const Tensor & batch2) const {
    return type().addbmm(beta, *this, batch1, batch2);
}
inline Tensor Tensor::addbmm(const Tensor & batch1, const Tensor & batch2) const {
    return type().addbmm(*this, batch1, batch2);
}
inline Tensor & Tensor::addbmm_(Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {
    return type().m_addbmm_(*this, beta, alpha, batch1, batch2);
}
inline Tensor & Tensor::addbmm_(Scalar beta, const Tensor & batch1, const Tensor & batch2) {
    return type().m_addbmm_(*this, beta, batch1, batch2);
}
inline Tensor & Tensor::addbmm_(const Tensor & batch1, const Tensor & batch2) {
    return type().m_addbmm_(*this, batch1, batch2);
}
inline Tensor Tensor::baddbmm(Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) const {
    return type().baddbmm(beta, *this, alpha, batch1, batch2);
}
inline Tensor Tensor::baddbmm(Scalar beta, const Tensor & batch1, const Tensor & batch2) const {
    return type().baddbmm(beta, *this, batch1, batch2);
}
inline Tensor Tensor::baddbmm(const Tensor & batch1, const Tensor & batch2) const {
    return type().baddbmm(*this, batch1, batch2);
}
inline Tensor & Tensor::baddbmm_(Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {
    return type().m_baddbmm_(*this, beta, alpha, batch1, batch2);
}
inline Tensor & Tensor::baddbmm_(Scalar beta, const Tensor & batch1, const Tensor & batch2) {
    return type().m_baddbmm_(*this, beta, batch1, batch2);
}
inline Tensor & Tensor::baddbmm_(const Tensor & batch1, const Tensor & batch2) {
    return type().m_baddbmm_(*this, batch1, batch2);
}
inline Tensor Tensor::addcmul(Scalar value, const Tensor & tensor1, const Tensor & tensor2) const {
    return type().addcmul(*this, value, tensor1, tensor2);
}
inline Tensor Tensor::addcmul(const Tensor & tensor1, const Tensor & tensor2) const {
    return type().addcmul(*this, tensor1, tensor2);
}
inline Tensor & Tensor::addcmul_(Scalar value, const Tensor & tensor1, const Tensor & tensor2) {
    return type().m_addcmul_(*this, value, tensor1, tensor2);
}
inline Tensor & Tensor::addcmul_(const Tensor & tensor1, const Tensor & tensor2) {
    return type().m_addcmul_(*this, tensor1, tensor2);
}
inline Tensor Tensor::addcdiv(Scalar value, const Tensor & tensor1, const Tensor & tensor2) const {
    return type().addcdiv(*this, value, tensor1, tensor2);
}
inline Tensor Tensor::addcdiv(const Tensor & tensor1, const Tensor & tensor2) const {
    return type().addcdiv(*this, tensor1, tensor2);
}
inline Tensor & Tensor::addcdiv_(Scalar value, const Tensor & tensor1, const Tensor & tensor2) {
    return type().m_addcdiv_(*this, value, tensor1, tensor2);
}
inline Tensor & Tensor::addcdiv_(const Tensor & tensor1, const Tensor & tensor2) {
    return type().m_addcdiv_(*this, tensor1, tensor2);
}
inline std::tuple<Tensor,Tensor> Tensor::gesv(const Tensor & A) const {
    return type().gesv(*this, A);
}
inline std::tuple<Tensor,Tensor> Tensor::gels(const Tensor & A) const {
    return type().gels(*this, A);
}
inline std::tuple<Tensor,Tensor> Tensor::trtrs(const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    return type().trtrs(*this, A, upper, transpose, unitriangular);
}
inline std::tuple<Tensor,Tensor> Tensor::trtrs(const Tensor & A, bool upper, bool transpose) const {
    return type().trtrs(*this, A, upper, transpose);
}
inline std::tuple<Tensor,Tensor> Tensor::trtrs(const Tensor & A, bool upper) const {
    return type().trtrs(*this, A, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::trtrs(const Tensor & A) const {
    return type().trtrs(*this, A);
}
inline std::tuple<Tensor,Tensor> Tensor::symeig(bool eigenvectors, bool upper) const {
    return type().symeig(*this, eigenvectors, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::symeig(bool eigenvectors) const {
    return type().symeig(*this, eigenvectors);
}
inline std::tuple<Tensor,Tensor> Tensor::symeig() const {
    return type().symeig(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::eig(bool eigenvectors) const {
    return type().eig(*this, eigenvectors);
}
inline std::tuple<Tensor,Tensor> Tensor::eig() const {
    return type().eig(*this);
}
inline std::tuple<Tensor,Tensor,Tensor> Tensor::svd(bool some) const {
    return type().svd(*this, some);
}
inline std::tuple<Tensor,Tensor,Tensor> Tensor::svd() const {
    return type().svd(*this);
}
inline Tensor Tensor::inverse() const {
    return type().inverse(*this);
}
inline Tensor Tensor::potrf(bool upper) const {
    return type().potrf(*this, upper);
}
inline Tensor Tensor::potrf() const {
    return type().potrf(*this);
}
inline Tensor Tensor::potrs(const Tensor & input2, bool upper) const {
    return type().potrs(*this, input2, upper);
}
inline Tensor Tensor::potrs(const Tensor & input2) const {
    return type().potrs(*this, input2);
}
inline Tensor Tensor::potri(bool upper) const {
    return type().potri(*this, upper);
}
inline Tensor Tensor::potri() const {
    return type().potri(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::pstrf(bool upper, Scalar tol) const {
    return type().pstrf(*this, upper, tol);
}
inline std::tuple<Tensor,Tensor> Tensor::pstrf(bool upper) const {
    return type().pstrf(*this, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::pstrf(Scalar tol) const {
    return type().pstrf(*this, tol);
}
inline std::tuple<Tensor,Tensor> Tensor::pstrf() const {
    return type().pstrf(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::qr() const {
    return type().qr(*this);
}
inline std::tuple<Tensor,Tensor> Tensor::geqrf() const {
    return type().geqrf(*this);
}
inline std::tuple<Tensor,const Tensor &> Tensor::orgqr(const Tensor & input2) const {
    return type().orgqr(*this, input2);
}
inline std::tuple<Tensor,const Tensor &> Tensor::ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    return type().ormqr(*this, input2, input3, left, transpose);
}
inline std::tuple<Tensor,const Tensor &> Tensor::ormqr(const Tensor & input2, const Tensor & input3, bool left) const {
    return type().ormqr(*this, input2, input3, left);
}
inline std::tuple<Tensor,const Tensor &> Tensor::ormqr(const Tensor & input2, const Tensor & input3) const {
    return type().ormqr(*this, input2, input3);
}
inline std::tuple<Tensor,Tensor> Tensor::btrifact(const Tensor & info) const {
    return type().btrifact(info, *this);
}
inline std::tuple<Tensor,Tensor> Tensor::btrifact() const {
    return type().btrifact(*this);
}
inline Tensor Tensor::btrisolve(const Tensor & LU_data, const Tensor & LU_pivots) const {
    return type().btrisolve(*this, LU_data, LU_pivots);
}
inline Tensor Tensor::multinomial(Generator & generator, int64_t num_samples, bool replacement) const {
    return type().multinomial(generator, *this, num_samples, replacement);
}
inline Tensor Tensor::multinomial(Generator & generator, int64_t num_samples) const {
    return type().multinomial(generator, *this, num_samples);
}
inline Tensor Tensor::multinomial(int64_t num_samples, bool replacement) const {
    return type().multinomial(*this, num_samples, replacement);
}
inline Tensor Tensor::multinomial(int64_t num_samples) const {
    return type().multinomial(*this, num_samples);
}
inline Tensor & Tensor::uniform_(Generator & generator, double from, double to) {
    return type().m_uniform_(*this, generator, from, to);
}
inline Tensor & Tensor::uniform_(Generator & generator, double from) {
    return type().m_uniform_(*this, generator, from);
}
inline Tensor & Tensor::uniform_(double from, double to) {
    return type().m_uniform_(*this, from, to);
}
inline Tensor & Tensor::uniform_(Generator & generator) {
    return type().m_uniform_(*this, generator);
}
inline Tensor & Tensor::uniform_(double from) {
    return type().m_uniform_(*this, from);
}
inline Tensor & Tensor::uniform_() {
    return type().m_uniform_(*this);
}
inline Tensor & Tensor::cauchy_(Generator & generator, double median, double sigma) {
    return type().m_cauchy_(*this, generator, median, sigma);
}
inline Tensor & Tensor::cauchy_(Generator & generator, double median) {
    return type().m_cauchy_(*this, generator, median);
}
inline Tensor & Tensor::cauchy_(double median, double sigma) {
    return type().m_cauchy_(*this, median, sigma);
}
inline Tensor & Tensor::cauchy_(Generator & generator) {
    return type().m_cauchy_(*this, generator);
}
inline Tensor & Tensor::cauchy_(double median) {
    return type().m_cauchy_(*this, median);
}
inline Tensor & Tensor::cauchy_() {
    return type().m_cauchy_(*this);
}
inline Tensor & Tensor::log_normal_(Generator & generator, double mean, double std) {
    return type().m_log_normal_(*this, generator, mean, std);
}
inline Tensor & Tensor::log_normal_(Generator & generator, double mean) {
    return type().m_log_normal_(*this, generator, mean);
}
inline Tensor & Tensor::log_normal_(double mean, double std) {
    return type().m_log_normal_(*this, mean, std);
}
inline Tensor & Tensor::log_normal_(Generator & generator) {
    return type().m_log_normal_(*this, generator);
}
inline Tensor & Tensor::log_normal_(double mean) {
    return type().m_log_normal_(*this, mean);
}
inline Tensor & Tensor::log_normal_() {
    return type().m_log_normal_(*this);
}
inline Tensor & Tensor::geometric_(Generator & generator, double p) {
    return type().m_geometric_(*this, generator, p);
}
inline Tensor & Tensor::geometric_(double p) {
    return type().m_geometric_(*this, p);
}
inline int64_t Tensor::size(int64_t dim) const {
    return type().m_size(*this, dim);
}
inline int64_t Tensor::stride(int64_t dim) const {
    return type().m_stride(*this, dim);
}
inline Tensor Tensor::select(int dim, int64_t sliceIndex) const {
    return type().select(*this, dim, sliceIndex);
}

template<typename T>
inline T* Tensor::data() const {
  runtime_error("data() cast to unexpected type.");
}
#define DEFINE_CAST(T,name,_) \
template<> \
inline T* Tensor::data() const { \
  AT_ASSERT(type().scalarType() == ScalarType::name, \
    "expected scalar type % s but found %s", #name, \
    at::toString(type().scalarType())); \
  return static_cast<T*>(this->data_ptr()); \
}

AT_FORALL_SCALAR_TYPES(DEFINE_CAST)
#undef DEFINE_CAST

} //namespace at

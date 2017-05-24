#pragma once

#include <THS/THS.h>

// We're defining THSTensor as a custom class
#undef THSTensor
#define THSRealTensor TH_CONCAT_3(THS,Real,Tensor)

#include "../Tensor.hpp"
#include "../Traits.hpp"

namespace thpp {

template<typename real>
struct ths_tensor_traits {};

#include "tensors/generic/THSTensor.hpp"
#include <THS/THSGenerateAllTypes.h>

} // namespace thpp

namespace thpp {

template<typename real>
struct THSTensor : public interface_traits<real>::tensor_interface_type {
  template<typename U>
  friend struct THSTensor;

  template<typename U>
  friend struct THTensor;

private:
  using interface_type = typename interface_traits<real>::tensor_interface_type;
public:
  using tensor_type = typename ths_tensor_traits<real>::tensor_type;
  using scalar_type = typename interface_type::scalar_type;
  using long_range = Tensor::long_range;

  THSTensor();
  THSTensor(tensor_type *wrapped);
  virtual ~THSTensor();

  virtual THSTensor* clone() const override;
  virtual THSTensor* clone_shallow() override;
  virtual std::unique_ptr<Tensor> contiguous() const override;
  virtual THSTensor* newSelect(int dimension, long sliceIndex) const override;
  virtual THSTensor* newNarrow(int dimension, long firstIndex, long size) const override;
  virtual THSTensor* newTranspose(int dimension1, int dimension2) const override;
  virtual THSTensor* newUnfold(int dimension, long size, long step) const override;
  virtual THSTensor* newExpand(const long_range& size) const override;
  virtual THSTensor* newView(const long_range& size) const override;

  virtual int nDim() const override;
  virtual long_range sizes() const override;
  virtual long_range strides() const override;
  virtual const long* rawSizes() const override;
  virtual const long* rawStrides() const override;
  virtual std::size_t storageOffset() const override;
  virtual std::size_t elementSize() const override;
  virtual long long numel() const override;
  virtual bool isContiguous() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual void* cdata() override;
  virtual const void* cdata() const override;
  virtual THSTensor& retain() override;
  virtual THSTensor& free() override;

  virtual THSTensor& resize(const std::initializer_list<long>& new_size) override;
  virtual THSTensor& resize(const std::vector<long>& new_size) override;
  virtual THSTensor& resize(THLongStorage *size,
                            THLongStorage *stride) override;
  virtual THSTensor& resizeAs(const Tensor& src) override;
  virtual THSTensor& set(const Tensor& src) override;
  virtual THSTensor& setStorage(const Storage& storage,
                                ptrdiff_t storageOffset,
                                const long_range& size,
                                const long_range& stride) override;
  virtual THSTensor& setStorage(const Storage& storage,
                                ptrdiff_t storageOffset,
                                THLongStorage *size,
                                THLongStorage *stride) override;

  virtual THSTensor& narrow(const Tensor& src, int dimension,
                           long firstIndex, long size) override;
  virtual THSTensor& select(const Tensor& src, int dimension,
                            long sliceIndex) override;
  virtual THSTensor& transpose(const Tensor& src, int dimension1,
                               int dimension2) override;
  virtual THSTensor& unfold(const Tensor& src, int dimension,
                            long size, long step) override;
  virtual THSTensor& squeeze(const Tensor& src) override;
  virtual THSTensor& squeeze(const Tensor& src, int dimension) override;
  virtual THSTensor& unsqueeze(const Tensor& src, int dimension) override;

  virtual THSTensor& gesv(const Tensor& ra, const Tensor& b, const Tensor& a) override;
  virtual THSTensor& trtrs(const Tensor& ra, const Tensor& b, const Tensor& a,
                           const char *uplo, const char *trans, const char *diag) override;
  virtual THSTensor& gels(const Tensor& ra, const Tensor& b, const Tensor& a) override;
  virtual THSTensor& syev(const Tensor& rv, const Tensor& a,
                          const char *jobz, const char *uplo) override;
  virtual THSTensor& geev(const Tensor& rv, const Tensor& a, const char *jobvr) override;
  virtual THSTensor& gesvd(const Tensor& rs, const Tensor& rv,
                           const Tensor& a, const char *jobu) override;
  virtual THSTensor& gesvd2(const Tensor& rs, const Tensor& rv, const Tensor& ra,
                            const Tensor& a, const char *jobu) override;
  virtual THSTensor& getri(const Tensor& a) override;
  virtual THSTensor& potrf(const Tensor& a, const char *uplo) override;
  virtual THSTensor& potrs(const Tensor& b, const Tensor& a, const char *uplo) override;
  virtual THSTensor& potri(const Tensor& a, const char *uplo) override;
  virtual THSTensor& qr(const Tensor& rr, const Tensor& a) override;
  virtual THSTensor& geqrf(const Tensor& rtau, const Tensor& a) override;
  virtual THSTensor& orgqr(const Tensor& a, const Tensor& tau) override;
  virtual THSTensor& ormqr(const Tensor& a, const Tensor& tau, const Tensor& c,
                           const char *side, const char *trans) override;
  virtual THSTensor& pstrf(const Tensor& rpiv, const Tensor& a,
                           const char *uplo, scalar_type tol) override;

  virtual THSTensor& diag(const Tensor& src, int k) override;
  virtual THSTensor& eye(long n, long m) override;
  virtual THSTensor& range(scalar_type xmin, scalar_type xmax,
                          scalar_type step) override;
  virtual THSTensor& randperm(const Generator& _generator, long n) override;
  virtual THSTensor& sort(const Tensor& ri, const Tensor& src,
                          int dimension, int desc) override;
  virtual THSTensor& topk(const Tensor& ri, const Tensor& src,
                          long k, int dim, int dir, int sorted) override;
  virtual THSTensor& tril(const Tensor& src, long k) override;
  virtual THSTensor& triu(const Tensor& src, long k) override;
  // TODO: remove in favor of cat
  virtual THSTensor& catArray(const std::vector<Tensor*>& inputs,
                              int dimension) override;
  virtual int equal(const Tensor& other) const override;

  // Note: the order in *Value and *Tensor is reversed compared to
  // the declarations in TH/generic/THTensorMath.h, so for instance
  // the call THRealTensor_ltTensor(r, ta, tb) is equivalent to
  // ta->ltTensor(r, tb). It is done this way so that the first
  // argument can be casted onto a byte tensor type
  virtual THSTensor& ltTensor(const Tensor& r, const Tensor& tb) override;
  virtual THSTensor& leTensor(const Tensor& r, const Tensor& tb) override;
  virtual THSTensor& gtTensor(const Tensor& r, const Tensor& tb) override;
  virtual THSTensor& geTensor(const Tensor& r, const Tensor& tb) override;
  virtual THSTensor& neTensor(const Tensor& r, const Tensor& tb) override;
  virtual THSTensor& eqTensor(const Tensor& r, const Tensor& tb) override;
  virtual THSTensor& ltTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THSTensor& leTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THSTensor& gtTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THSTensor& geTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THSTensor& neTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THSTensor& eqTensorT(const Tensor& ta, const Tensor& tb) override;

  virtual THSTensor& abs(const Tensor& src) override;
  virtual THSTensor& sigmoid(const Tensor& src) override;
  virtual THSTensor& log(const Tensor& src) override;
  virtual THSTensor& log1p(const Tensor& src) override;
  virtual THSTensor& exp(const Tensor& src) override;
  virtual THSTensor& cos(const Tensor& src) override;
  virtual THSTensor& acos(const Tensor& src) override;
  virtual THSTensor& cosh(const Tensor& src) override;
  virtual THSTensor& sin(const Tensor& src) override;
  virtual THSTensor& asin(const Tensor& src) override;
  virtual THSTensor& sinh(const Tensor& src) override;

  virtual THSTensor& tan(const Tensor& src) override;
  virtual THSTensor& atan(const Tensor& src) override;
  virtual THSTensor& atan2(const Tensor& src1, const Tensor& src2) override;
  virtual THSTensor& tanh(const Tensor& src) override;
  virtual THSTensor& pow(const Tensor& src, scalar_type value) override;
  virtual THSTensor& tpow(scalar_type value, const Tensor& src) override;
  virtual THSTensor& sqrt(const Tensor& src) override;
  virtual THSTensor& rsqrt(const Tensor& src) override;
  virtual THSTensor& ceil(const Tensor& src) override;
  virtual THSTensor& floor(const Tensor& src) override;
  virtual THSTensor& round(const Tensor& src) override;
  virtual THSTensor& trunc(const Tensor& src) override;
  virtual THSTensor& frac(const Tensor& src) override;
  virtual THSTensor& lerp(const Tensor& a, const Tensor& b, scalar_type weight) override;
  virtual THSTensor& mean(const Tensor& src, int dimension, int keepdim) override;
  virtual THSTensor& std(const Tensor& src, int dimension, int flag, int keepdim) override;
  virtual THSTensor& var(const Tensor& src, int dimension, int flag, int keepdim) override;
  virtual THSTensor& norm(const Tensor& src, scalar_type value, int dimension, int keepdim) override;
  virtual THSTensor& renorm(const Tensor& src, scalar_type value, int dimension, scalar_type maxnorm) override;
  virtual THSTensor& histc(const Tensor& src, long nbins, scalar_type minvalue, scalar_type maxvalue) override;
  virtual THSTensor& bhistc(const Tensor& src, long nbins, scalar_type minvalue, scalar_type maxvalue) override;
  virtual scalar_type dist(const Tensor& src, scalar_type value) override;
  virtual scalar_type meanall() override;
  virtual scalar_type varall() override;
  virtual scalar_type stdall() override;
  virtual scalar_type normall(scalar_type value) override;
  virtual THSTensor& linspace(scalar_type a, scalar_type b, long n) override;
  virtual THSTensor& logspace(scalar_type a, scalar_type b, long n) override;
  virtual THSTensor& rand(const Generator& _generator, THLongStorage *size) override;
  virtual THSTensor& randn(const Generator& _generator, THLongStorage *size) override;
  virtual int logicalall() override;
  virtual int logicalany() override;
  virtual THSTensor& random(const Generator& _generator) override;
  virtual THSTensor& geometric(const Generator& _generator, double p) override;
  virtual THSTensor& bernoulli(const Generator& _generator, double p) override;
  virtual THSTensor& bernoulli_FloatTensor(const Generator& _generator, const Tensor& p) override;
  virtual THSTensor& bernoulli_DoubleTensor(const Generator& _generator, const Tensor& p) override;
  virtual THSTensor& uniform(const Generator& _generator, double a, double b) override;
  virtual THSTensor& normal(const Generator& _generator, double mean, double stdv) override;
  virtual THSTensor& exponential(const Generator& _generator, double lambda) override;
  virtual THSTensor& cauchy(const Generator& _generator, double median, double sigma) override;
  virtual THSTensor& logNormal(const Generator& _generator, double mean, double stdv) override;

  // Note: the order of *Tensor and *Prob_dist is reversed compared to
  // the declarations in TH/generic/THTensorMath.h, so for instance
  // the call:
  // THRealTensor_multinomial(r, _generator, prob_dist, n_sample, with_replacement)
  // is equivalent to `prob_dist->multinomial(r, _generator, n_sample, with_replacement)`.
  // It is done this way so that the first argument can be casted onto a float tensor type.
  virtual THSTensor& multinomial(const Tensor& r, const Generator& _generator,
                                 int n_sample, int with_replacement) override;

  virtual THSTensor& ltValue(const Tensor& t, scalar_type value) override;
  virtual THSTensor& leValue(const Tensor& t, scalar_type value) override;
  virtual THSTensor& gtValue(const Tensor& t, scalar_type value) override;
  virtual THSTensor& geValue(const Tensor& t, scalar_type value) override;
  virtual THSTensor& neValue(const Tensor& t, scalar_type value) override;
  virtual THSTensor& eqValue(const Tensor& t, scalar_type value) override;
  virtual THSTensor& ltValueT(const Tensor& t, scalar_type value) override;
  virtual THSTensor& leValueT(const Tensor& t, scalar_type value) override;
  virtual THSTensor& gtValueT(const Tensor& t, scalar_type value) override;
  virtual THSTensor& geValueT(const Tensor& t, scalar_type value) override;
  virtual THSTensor& neValueT(const Tensor& t, scalar_type value) override;
  virtual THSTensor& eqValueT(const Tensor& t, scalar_type value) override;

  virtual THSTensor& fill(scalar_type value) override;
  virtual THSTensor& maskedFill(const Tensor& mask, scalar_type value) override;
  virtual THSTensor& maskedCopy(const Tensor& mask, const Tensor& src) override;
  virtual THSTensor& maskedSelect(const Tensor& mask, const Tensor& src) override;
  // NOTE like in byte comparison operations, the order in nonzero
  // is reversed compared to THS, i.e. tensor->nonzero(subscript) is equivalent
  // to THSTensor_(nonzero)(subscript, tensor)
  virtual THSTensor& nonzero(const Tensor& subscript) override;
  virtual THSTensor& indexSelect(const Tensor& src, int dim, const Tensor& index) override;
  virtual THSTensor& indexCopy(int dim, const Tensor& index, const Tensor& src) override;
  virtual THSTensor& indexAdd(int dim, const Tensor& index, const Tensor& src) override;
  virtual THSTensor& indexFill(int dim, const Tensor& index, scalar_type value) override;

  virtual THSTensor& copy(const Tensor& src) override;
  virtual THSTensor& cat(const std::vector<Tensor*>& src, int dimension) override;
  virtual THSTensor& gather(const Tensor& src, int dimension, const Tensor& index) override;
  virtual THSTensor& scatter(int dimension, const Tensor& index, const Tensor& src) override;
  virtual THSTensor& scatterFill(int dimension, const Tensor& index, scalar_type value) override;
  virtual scalar_type dot(const Tensor& source) override;
  virtual scalar_type minall() override;
  virtual scalar_type maxall() override;
  virtual scalar_type sumall() override;
  virtual scalar_type prodall() override;
  virtual THSTensor& neg(const Tensor& src) override;
  virtual THSTensor& cinv(const Tensor& src) override;
  virtual THSTensor& add(const Tensor& src, scalar_type value) override;
  virtual THSTensor& sub(const Tensor& src, scalar_type value) override;
  virtual THSTensor& mul(const Tensor& src, scalar_type value) override;
  virtual THSTensor& div(const Tensor& src, scalar_type value) override;
  virtual THSTensor& fmod(const Tensor& src, scalar_type value) override;
  virtual THSTensor& remainder(const Tensor& src, scalar_type value) override;
  virtual THSTensor& clamp(const Tensor& src, scalar_type min_value, scalar_type max_value) override;
  virtual THSTensor& cadd(const Tensor& src1, const Tensor& src2) override;
  virtual THSTensor& cadd(const Tensor& src1, scalar_type value, const Tensor& src2) override;
  virtual THSTensor& csub(const Tensor& src1, scalar_type value, const Tensor& src2) override;
  virtual THSTensor& cmul(const Tensor& src1, const Tensor& src2) override;
  virtual THSTensor& cpow(const Tensor& src1, const Tensor& src2) override;
  virtual THSTensor& cdiv(const Tensor& src1, const Tensor& src2) override;
  virtual THSTensor& cfmod(const Tensor& src1, const Tensor& src2) override;
  virtual THSTensor& cremainder(const Tensor& src1, const Tensor& src2) override;
  virtual THSTensor& addcmul(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) override;
  virtual THSTensor& addcdiv(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) override;
  virtual THSTensor& addmv(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat, const Tensor& vec) override;
  virtual THSTensor& addmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat1, const Tensor& mat2) override;
  virtual THSTensor& addr(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& vec1, const Tensor& vec2) override;
  virtual THSTensor& addbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) override;
  virtual THSTensor& baddbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) override;
  virtual THSTensor& match(const Tensor& m1, const Tensor& m2, scalar_type gain) override;
  virtual THSTensor& max(const Tensor& indices_, const Tensor& src, int dimension, int keepdim) override;
  virtual THSTensor& min(const Tensor& indices_, const Tensor& src, int dimension, int keepdim) override;
  virtual THSTensor& kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension, int keepdim) override;
  virtual THSTensor& mode(const Tensor& indices_, const Tensor& src, int dimension, int keepdim) override;
  virtual THSTensor& median(const Tensor& indices_, const Tensor& src, int dimension, int keepdim) override;
  virtual THSTensor& sum(const Tensor& src, int dimension, int keepdim) override;
  virtual THSTensor& prod(const Tensor& src, int dimension, int keepdim) override;
  virtual THSTensor& cumsum(const Tensor& src, int dimension) override;
  virtual THSTensor& cumprod(const Tensor& src, int dimension) override;
  virtual THSTensor& sign(const Tensor& source) override;
  virtual scalar_type trace() override;
  virtual THSTensor& cross(const Tensor& src1, const Tensor& src2, int dimension) override;
  virtual THSTensor& cmax(const Tensor& src1, const Tensor& src2) override;
  virtual THSTensor& cmin(const Tensor& src1, const Tensor& src2) override;
  virtual THSTensor& cmaxValue(const Tensor& src, scalar_type value) override;
  virtual THSTensor& cminValue(const Tensor& src, scalar_type value) override;
  virtual THSTensor& zero() override;

  virtual thpp::Type type() const override;
  virtual bool isCuda() const override;
  virtual bool isSparse() const override;
  virtual int getDevice() const override;
  virtual std::unique_ptr<Tensor> newTensor() const override;

private:
  template<typename iterator>
  THSTensor& resize(const iterator& begin, const iterator& end);
  template<typename iterator>
  THSTensor& resize(const iterator& size_begin, const iterator& size_end,
                    const iterator& stride_begin, const iterator& stride_end);

protected:
  tensor_type *tensor;
};

}

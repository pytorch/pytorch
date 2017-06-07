#pragma once

#include <THC/THC.h>

// We're defining THCTensor as a custom class
#undef THCTensor
#define THCRealTensor TH_CONCAT_3(TH,CReal,Tensor)

#include "../Tensor.hpp"
#include "../Traits.hpp"

namespace thpp {

template<typename real>
struct thc_tensor_traits {};

#include "tensors/generic/THCTensor.hpp"
#include <THC/THCGenerateAllTypes.h>

} // namespace thpp

#include "../storages/THCStorage.hpp"

namespace thpp {

template<typename real>
struct THCTensor : public interface_traits<real>::tensor_interface_type {
  template<typename U>
  friend struct THCTensor;

private:
  using interface_type = typename interface_traits<real>::tensor_interface_type;
public:
  using tensor_type = typename thc_tensor_traits<real>::tensor_type;
  using scalar_type = typename interface_type::scalar_type;
  using long_range = Tensor::long_range;

  THCTensor(THCState* state);
  THCTensor(THCState* state, tensor_type *wrapped);
  virtual ~THCTensor();

  virtual THCTensor* clone() const override;
  virtual THCTensor* clone_shallow() override;
  virtual std::unique_ptr<Tensor> contiguous() const override;
  virtual THCTensor* newSelect(int dimension, long sliceIndex) const override;
  virtual THCTensor* newNarrow(int dimension, long firstIndex, long size) const override;
  virtual THCTensor* newTranspose(int dimension1, int dimension2) const override;
  virtual THCTensor* newUnfold(int dimension, long size, long step) const override;
  virtual THCTensor* newExpand(const long_range& size) const override;
  virtual THCTensor* newView(const long_range& size) const override;

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
  virtual THCTensor& retain() override;
  virtual THCTensor& free() override;

  virtual THCTensor& resize(const std::initializer_list<long>& new_size) override;
  virtual THCTensor& resize(const std::vector<long>& new_size) override;
  virtual THCTensor& resize(THLongStorage *size,
                            THLongStorage *stride) override;
  virtual THCTensor& resizeAs(const Tensor& src) override;
  virtual THCTensor& set(const Tensor& src) override;
  virtual THCTensor& setStorage(const Storage& storage,
                                ptrdiff_t storageOffset,
                                const long_range& size,
                                const long_range& stride) override;
  virtual THCTensor& setStorage(const Storage& storage,
                                ptrdiff_t storageOffset,
                                THLongStorage *size,
                                THLongStorage *stride) override;

  virtual THCTensor& narrow(const Tensor& src, int dimension,
                           long firstIndex, long size) override;
  virtual THCTensor& select(const Tensor& src, int dimension,
                           long sliceIndex) override;
  virtual THCTensor& transpose(const Tensor& src, int dimension1,
                               int dimension2) override;
  virtual THCTensor& unfold(const Tensor& src, int dimension,
                           long size, long step) override;
  virtual THCTensor& squeeze(const Tensor& src) override;
  virtual THCTensor& squeeze(const Tensor& src, int dimension) override;
  virtual THCTensor& unsqueeze(const Tensor& src, int dimension) override;

  virtual THCTensor& gesv(const Tensor& ra, const Tensor& b, const Tensor& a) override;
  virtual THCTensor& trtrs(const Tensor& ra, const Tensor& b, const Tensor& a,
                           const char *uplo, const char *trans, const char *diag) override;
  virtual THCTensor& gels(const Tensor& ra, const Tensor& b, const Tensor& a) override;
  virtual THCTensor& syev(const Tensor& rv, const Tensor& a,
                          const char *jobz, const char *uplo) override;
  virtual THCTensor& geev(const Tensor& rv, const Tensor& a, const char *jobvr) override;
  virtual THCTensor& gesvd(const Tensor& rs, const Tensor& rv,
                           const Tensor& a, const char *jobu) override;
  virtual THCTensor& gesvd2(const Tensor& rs, const Tensor& rv, const Tensor& ra,
                            const Tensor& a, const char *jobu) override;
  virtual THCTensor& getri(const Tensor& a) override;
  virtual THCTensor& potrf(const Tensor& a, const char *uplo) override;
  virtual THCTensor& potrs(const Tensor& b, const Tensor& a, const char *uplo) override;
  virtual THCTensor& potri(const Tensor& a, const char *uplo) override;
  virtual THCTensor& qr(const Tensor& rr, const Tensor& a) override;
  virtual THCTensor& geqrf(const Tensor& rtau, const Tensor& a) override;
  virtual THCTensor& orgqr(const Tensor& a, const Tensor& tau) override;
  virtual THCTensor& ormqr(const Tensor& a, const Tensor& tau, const Tensor& c,
                           const char *side, const char *trans) override;
  virtual THCTensor& pstrf(const Tensor& rpiv, const Tensor& a,
                           const char *uplo, scalar_type tol) override;

  virtual THCTensor& diag(const Tensor& src, int k) override;
  virtual THCTensor& eye(long n, long m) override;
  virtual THCTensor& range(scalar_type xmin, scalar_type xmax,
                          scalar_type step) override;
  virtual THCTensor& randperm(const Generator& _generator, long n) override;
  virtual THCTensor& sort(const Tensor& ri, const Tensor& src,
                       int dimension, int desc) override;
  virtual THCTensor& topk(const Tensor& ri, const Tensor& src,
                       long k, int dim, int dir, int sorted) override;
  virtual THCTensor& tril(const Tensor& src, long k) override;
  virtual THCTensor& triu(const Tensor& src, long k) override;
  // TODO: remove in favor of cat
  virtual THCTensor& catArray(const std::vector<Tensor*>& inputs,
                             int dimension) override;
  virtual int equal(const Tensor& other) const override;

  // Note: the order in *Value and *Tensor is reversed compared to
  // the declarations in TH/generic/THTensorMath.h, so for instance
  // the call THRealTensor_ltTensor(r, ta, tb) is equivalent to
  // ta->ltTensor(r, tb). It is done this way so that the first
  // argument can be casted onto a byte tensor type
  virtual THCTensor& ltTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCTensor& leTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCTensor& gtTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCTensor& geTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCTensor& neTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCTensor& eqTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCTensor& ltTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THCTensor& leTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THCTensor& gtTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THCTensor& geTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THCTensor& neTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THCTensor& eqTensorT(const Tensor& ta, const Tensor& tb) override;

  virtual THCTensor& abs(const Tensor& src) override;
  virtual THCTensor& sigmoid(const Tensor& src) override;
  virtual THCTensor& log(const Tensor& src) override;
  virtual THCTensor& log1p(const Tensor& src) override;
  virtual THCTensor& exp(const Tensor& src) override;
  virtual THCTensor& cos(const Tensor& src) override;
  virtual THCTensor& acos(const Tensor& src) override;
  virtual THCTensor& cosh(const Tensor& src) override;
  virtual THCTensor& sin(const Tensor& src) override;
  virtual THCTensor& asin(const Tensor& src) override;
  virtual THCTensor& sinh(const Tensor& src) override;

  virtual THCTensor& tan(const Tensor& src) override;
  virtual THCTensor& atan(const Tensor& src) override;
  virtual THCTensor& atan2(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& tanh(const Tensor& src) override;
  virtual THCTensor& pow(const Tensor& src, scalar_type value) override;
  virtual THCTensor& tpow(scalar_type value, const Tensor& src) override;
  virtual THCTensor& sqrt(const Tensor& src) override;
  virtual THCTensor& rsqrt(const Tensor& src) override;
  virtual THCTensor& ceil(const Tensor& src) override;
  virtual THCTensor& floor(const Tensor& src) override;
  virtual THCTensor& round(const Tensor& src) override;
  virtual THCTensor& trunc(const Tensor& src) override;
  virtual THCTensor& frac(const Tensor& src) override;
  virtual THCTensor& lerp(const Tensor& a, const Tensor& b, scalar_type weight) override;
  virtual THCTensor& mean(const Tensor& src, int dimension, int keepdim) override;
  virtual THCTensor& std(const Tensor& src, int dimension, int flag, int keepdim) override;
  virtual THCTensor& var(const Tensor& src, int dimension, int flag, int keepdim) override;
  virtual THCTensor& norm(const Tensor& src, scalar_type value, int dimension, int keepdim) override;
  virtual THCTensor& renorm(const Tensor& src, scalar_type value, int dimension, scalar_type maxnorm) override;
  virtual THCTensor& histc(const Tensor& src, long nbins, scalar_type minvalue, scalar_type maxvalue) override;
  virtual THCTensor& bhistc(const Tensor& src, long nbins, scalar_type minvalue, scalar_type maxvalue) override;
  virtual scalar_type dist(const Tensor& src, scalar_type value) override;
  virtual scalar_type meanall() override;
  virtual scalar_type varall() override;
  virtual scalar_type stdall() override;
  virtual scalar_type normall(scalar_type value) override;
  virtual THCTensor& linspace(scalar_type a, scalar_type b, long n) override;
  virtual THCTensor& logspace(scalar_type a, scalar_type b, long n) override;
  virtual THCTensor& rand(const Generator& _generator, THLongStorage *size) override;
  virtual THCTensor& randn(const Generator& _generator, THLongStorage *size) override;
  virtual int logicalall() override;
  virtual int logicalany() override;
  virtual THCTensor& random(const Generator& _generator) override;
  virtual THCTensor& geometric(const Generator& _generator, double p) override;
  virtual THCTensor& bernoulli(const Generator& _generator, double p) override;
  virtual THCTensor& bernoulli_FloatTensor(const Generator& _generator, const Tensor& p) override;
  virtual THCTensor& bernoulli_DoubleTensor(const Generator& _generator, const Tensor& p) override;
  virtual THCTensor& uniform(const Generator& _generator, double a, double b) override;
  virtual THCTensor& normal(const Generator& _generator, double mean, double stdv) override;
  virtual THCTensor& exponential(const Generator& _generator, double lambda) override;
  virtual THCTensor& cauchy(const Generator& _generator, double median, double sigma) override;
  virtual THCTensor& logNormal(const Generator& _generator, double mean, double stdv) override;

  // Note: the order of *Tensor and *Prob_dist is reversed compared to
  // the declarations in TH/generic/THTensorMath.h, so for instance
  // the call:
  // THRealTensor_multinomial(r, _generator, prob_dist, n_sample, with_replacement)
  // is equivalent to `prob_dist->multinomial(r, _generator, n_sample, with_replacement)`.
  // It is done this way so that the first argument can be casted onto a float tensor type.
  virtual THCTensor& multinomial(const Tensor& r, const Generator& _generator,
                                 int n_sample, int with_replacement) override;

  virtual THCTensor& ltValue(const Tensor& t, scalar_type value) override;
  virtual THCTensor& leValue(const Tensor& t, scalar_type value) override;
  virtual THCTensor& gtValue(const Tensor& t, scalar_type value) override;
  virtual THCTensor& geValue(const Tensor& t, scalar_type value) override;
  virtual THCTensor& neValue(const Tensor& t, scalar_type value) override;
  virtual THCTensor& eqValue(const Tensor& t, scalar_type value) override;
  virtual THCTensor& ltValueT(const Tensor& t, scalar_type value) override;
  virtual THCTensor& leValueT(const Tensor& t, scalar_type value) override;
  virtual THCTensor& gtValueT(const Tensor& t, scalar_type value) override;
  virtual THCTensor& geValueT(const Tensor& t, scalar_type value) override;
  virtual THCTensor& neValueT(const Tensor& t, scalar_type value) override;
  virtual THCTensor& eqValueT(const Tensor& t, scalar_type value) override;

  virtual THCTensor& fill(scalar_type value) override;
  virtual THCTensor& maskedFill(const Tensor& mask, scalar_type value) override;
  virtual THCTensor& maskedCopy(const Tensor& mask, const Tensor& src) override;
  virtual THCTensor& maskedSelect(const Tensor& mask, const Tensor& src) override;
  // NOTE like in byte comparison operations, the order in nonzero
  // is reversed compared to THC, i.e. tensor->nonzero(subscript) is equivalent
  // to THCTensor_(nonzero)(subscript, tensor)
  virtual THCTensor& nonzero(const Tensor& subscript) override;
  virtual THCTensor& indexSelect(const Tensor& src, int dim, const Tensor& index) override;
  virtual THCTensor& indexCopy(int dim, const Tensor& index, const Tensor& src) override;
  virtual THCTensor& indexAdd(int dim, const Tensor& index, const Tensor& src) override;
  virtual THCTensor& indexFill(int dim, const Tensor& index, scalar_type value) override;

  virtual THCTensor& copy(const Tensor& src) override;
  virtual THCTensor& cat(const std::vector<Tensor*>& src, int dimension) override;
  virtual THCTensor& gather(const Tensor& src, int dimension,
                           const Tensor& index) override;
  virtual THCTensor& scatter(int dimension, const Tensor& index,
                            const Tensor& src) override;
  virtual THCTensor& scatterFill(int dimension, const Tensor& index,
                                scalar_type value) override;
  virtual scalar_type dot(const Tensor& source) override;
  virtual scalar_type minall() override;
  virtual scalar_type maxall() override;
  virtual scalar_type medianall() override;
  virtual scalar_type sumall() override;
  virtual scalar_type prodall() override;
  virtual THCTensor& neg(const Tensor& src) override;
  virtual THCTensor& cinv(const Tensor& src) override;
  virtual THCTensor& add(const Tensor& src, scalar_type value) override;
  virtual THCTensor& sub(const Tensor& src, scalar_type value) override;
  virtual THCTensor& mul(const Tensor& src, scalar_type value) override;
  virtual THCTensor& div(const Tensor& src, scalar_type value) override;
  virtual THCTensor& fmod(const Tensor& src, scalar_type value) override;
  virtual THCTensor& remainder(const Tensor& src, scalar_type value) override;
  virtual THCTensor& clamp(const Tensor& src, scalar_type min_value,
                          scalar_type max_value) override;
  virtual THCTensor& cadd(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cadd(const Tensor& src1, scalar_type value,
                         const Tensor& src2) override;
  virtual THCTensor& csub(const Tensor& src1, scalar_type value,
                         const Tensor& src2) override;
  virtual THCTensor& cmul(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cpow(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cdiv(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cfmod(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cremainder(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& addcmul(const Tensor& src1, scalar_type value,
                            const Tensor& src2, const Tensor& src3) override;
  virtual THCTensor& addcdiv(const Tensor& src1, scalar_type value,
                            const Tensor& src2, const Tensor& src3) override;
  virtual THCTensor& addmv(scalar_type beta, const Tensor& src,
                          scalar_type alpha, const Tensor& mat,
                          const Tensor& vec) override;
  virtual THCTensor& addmm(scalar_type beta, const Tensor& src,
                          scalar_type alpha, const Tensor& mat1,
                          const Tensor& mat2) override;
  virtual THCTensor& addr(scalar_type beta, const Tensor& src,
                         scalar_type alpha, const Tensor& vec1,
                         const Tensor& vec2) override;
  virtual THCTensor& addbmm(scalar_type beta, const Tensor& src,
                           scalar_type alpha, const Tensor& batch1,
                           const Tensor& batch2) override;
  virtual THCTensor& baddbmm(scalar_type beta, const Tensor& src,
                            scalar_type alpha, const Tensor& batch1,
                            const Tensor& batch2) override;
  virtual THCTensor& match(const Tensor& m1, const Tensor& m2,
                          scalar_type gain) override;
  virtual THCTensor& max(const Tensor& indices_, const Tensor& src,
                        int dimension, int keepdim) override;
  virtual THCTensor& min(const Tensor& indices_, const Tensor& src,
                        int dimension, int keepdim) override;
  virtual THCTensor& kthvalue(const Tensor& indices_, const Tensor& src,
                             long k, int dimension, int keepdim) override;
  virtual THCTensor& mode(const Tensor& indices_, const Tensor& src,
                         int dimension, int keepdim) override;
  virtual THCTensor& median(const Tensor& indices_, const Tensor& src,
                           int dimension, int keepdim) override;
  virtual THCTensor& sum(const Tensor& src, int dimension, int keepdim) override;
  virtual THCTensor& prod(const Tensor& src, int dimension, int keepdim) override;
  virtual THCTensor& cumsum(const Tensor& src, int dimension) override;
  virtual THCTensor& cumprod(const Tensor& src, int dimension) override;
  virtual THCTensor& sign(const Tensor& source) override;
  virtual scalar_type trace() override;
  virtual THCTensor& cross(const Tensor& src1, const Tensor& src2,
                          int dimension) override;
  virtual THCTensor& cmax(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cmin(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cmaxValue(const Tensor& src, scalar_type value) override;
  virtual THCTensor& cminValue(const Tensor& src, scalar_type value) override;
  virtual THCTensor& zero() override;

  virtual thpp::Type type() const override;
  virtual bool isCuda() const override;
  virtual bool isSparse() const override;
  virtual int getDevice() const override;
  virtual std::unique_ptr<Tensor> newTensor() const override;

private:
  template<typename iterator>
  THCTensor& resize(const iterator& begin, const iterator& end);
  template<typename iterator>
  THCTensor& resize(const iterator& size_begin, const iterator& size_end,
                   const iterator& stride_begin, const iterator& stride_end);

protected:
  tensor_type *tensor;
  THCState *state;
};

}

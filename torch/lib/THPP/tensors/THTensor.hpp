#pragma once

#include <TH/TH.h>

// We're defining THTensor as a custom class
#undef THTensor
#define THRealTensor TH_CONCAT_3(TH,Real,Tensor)

#include "../Tensor.hpp"
#include "../Traits.hpp"

namespace thpp {

template<typename real>
struct th_tensor_traits {};

#include "tensors/generic/THTensor.hpp"
#include <TH/THGenerateAllTypes.h>

} // namespace thpp

#include "../storages/THStorage.hpp"

namespace thpp {

template<typename real>
struct THTensor : public interface_traits<real>::tensor_interface_type {
  template<typename U>
  friend class THTensor;

private:
  using interface_type = typename interface_traits<real>::tensor_interface_type;
public:
  using tensor_type = typename th_tensor_traits<real>::tensor_type;
  using scalar_type = typename interface_type::scalar_type;
  using long_range = Tensor::long_range;

  THTensor();
  THTensor(tensor_type *wrapped);
  virtual ~THTensor();

  virtual THTensor* clone() const override;
  virtual THTensor* clone_shallow() override;
  virtual std::unique_ptr<Tensor> contiguous() const override;

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
  virtual THTensor& retain() override;
  virtual THTensor& free() override;

  virtual THTensor& resize(const std::initializer_list<long>& new_size) override;
  virtual THTensor& resize(const std::vector<long>& new_size) override;
  virtual THTensor& resize(THLongStorage *size,
                           THLongStorage *stride) override;
  virtual THTensor& resizeAs(const Tensor& src) override;
  virtual THTensor& set(const Tensor& src) override;
  virtual THTensor& setStorage(const Storage& storage,
                               ptrdiff_t storageOffset,
                               const long_range& size,
                               const long_range& stride) override;
  virtual THTensor& setStorage(const Storage& storage,
                             ptrdiff_t storageOffset,
                             THLongStorage *size,
                             THLongStorage *stride) override;

  virtual THTensor& narrow(const Tensor& src, int dimension,
                           long firstIndex, long size) override;
  virtual THTensor& select(const Tensor& src, int dimension,
                           long sliceIndex) override;
  virtual THTensor& transpose(const Tensor& src, int dimension1,
                              int dimension2) override;
  virtual THTensor& unfold(const Tensor& src, int dimension,
                           long size, long step) override;
  virtual THTensor& squeeze(const Tensor& src, int dimension) override;
  virtual THTensor& unsqueeze(const Tensor& src, int dimension) override;

  virtual THTensor& diag(const Tensor& src, int k) override;
  virtual THTensor& eye(long n, long m) override;
  virtual THTensor& range(scalar_type xmin, scalar_type xmax,
                          scalar_type step) override;
  virtual THTensor& sort(const Tensor& ri, const Tensor& src,
                       int dimension, int desc) override;
  virtual THTensor& topk(const Tensor& ri, const Tensor& src,
                       long k, int dim, int dir, int sorted) override;
  virtual THTensor& tril(const Tensor& src, long k) override;
  virtual THTensor& triu(const Tensor& src, long k) override;
  // TODO: remove in favor of cat
  virtual THTensor& catArray(const std::vector<Tensor*>& inputs,
                             int dimension) override;
  virtual int equal(const Tensor& other) const override;

  // Note: the order in *Value and *Tensor is reversed compared to
  // the declarations in TH/generic/THTensorMath.h, so for instance
  // the call THRealTensor_ltTensor(r, ta, tb) is equivalent to
  // ta->ltTensor(r, tb). It is done this way so that the first
  // argument can be casted onto a byte tensor type
  virtual THTensor& ltTensor(const Tensor& r, const Tensor& tb) override;
  virtual THTensor& leTensor(const Tensor& r, const Tensor& tb) override;
  virtual THTensor& gtTensor(const Tensor& r, const Tensor& tb) override;
  virtual THTensor& geTensor(const Tensor& r, const Tensor& tb) override;
  virtual THTensor& neTensor(const Tensor& r, const Tensor& tb) override;
  virtual THTensor& eqTensor(const Tensor& r, const Tensor& tb) override;
  virtual THTensor& ltTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THTensor& leTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THTensor& gtTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THTensor& geTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THTensor& neTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THTensor& eqTensorT(const Tensor& ta, const Tensor& tb) override;

  virtual THTensor& abs(const Tensor& src) override;
  virtual THTensor& sigmoid(const Tensor& src) override;
  virtual THTensor& log(const Tensor& src) override;
  virtual THTensor& log1p(const Tensor& src) override;
  virtual THTensor& exp(const Tensor& src) override;
  virtual THTensor& cos(const Tensor& src) override;
  virtual THTensor& acos(const Tensor& src) override;
  virtual THTensor& cosh(const Tensor& src) override;
  virtual THTensor& sin(const Tensor& src) override;
  virtual THTensor& asin(const Tensor& src) override;
  virtual THTensor& sinh(const Tensor& src) override;

  virtual THTensor& ltValue(const Tensor& t, scalar_type value) override;
  virtual THTensor& leValue(const Tensor& t, scalar_type value) override;
  virtual THTensor& gtValue(const Tensor& t, scalar_type value) override;
  virtual THTensor& geValue(const Tensor& t, scalar_type value) override;
  virtual THTensor& neValue(const Tensor& t, scalar_type value) override;
  virtual THTensor& eqValue(const Tensor& t, scalar_type value) override;
  virtual THTensor& ltValueT(const Tensor& t, scalar_type value) override;
  virtual THTensor& leValueT(const Tensor& t, scalar_type value) override;
  virtual THTensor& gtValueT(const Tensor& t, scalar_type value) override;
  virtual THTensor& geValueT(const Tensor& t, scalar_type value) override;
  virtual THTensor& neValueT(const Tensor& t, scalar_type value) override;
  virtual THTensor& eqValueT(const Tensor& t, scalar_type value) override;
  virtual THTensor& fill(scalar_type value) override;

  virtual THTensor& copy(const Tensor& src) override;
  virtual THTensor& cat(const std::vector<Tensor*>& src, int dimension) override;
  virtual THTensor& gather(const Tensor& src, int dimension,
                           const Tensor& index) override;
  virtual THTensor& scatter(int dimension, const Tensor& index,
                            const Tensor& src) override;
  virtual THTensor& scatterFill(int dimension, const Tensor& index,
                                scalar_type value) override;
  virtual scalar_type dot(const Tensor& source) override;
  virtual scalar_type minall() override;
  virtual scalar_type maxall() override;
  virtual scalar_type sumall() override;
  virtual scalar_type prodall() override;
  virtual THTensor& neg(const Tensor& src) override;
  virtual THTensor& cinv(const Tensor& src) override;
  virtual THTensor& add(const Tensor& src, scalar_type value) override;
  virtual THTensor& sub(const Tensor& src, scalar_type value) override;
  virtual THTensor& mul(const Tensor& src, scalar_type value) override;
  virtual THTensor& div(const Tensor& src, scalar_type value) override;
  virtual THTensor& fmod(const Tensor& src, scalar_type value) override;
  virtual THTensor& remainder(const Tensor& src, scalar_type value) override;
  virtual THTensor& clamp(const Tensor& src, scalar_type min_value,
                          scalar_type max_value) override;
  virtual THTensor& cadd(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cadd(const Tensor& src1, scalar_type value,
                         const Tensor& src2) override;
  virtual THTensor& csub(const Tensor& src1, scalar_type value,
                         const Tensor& src2) override;
  virtual THTensor& cmul(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cpow(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cdiv(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cfmod(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cremainder(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& addcmul(const Tensor& src1, scalar_type value,
                            const Tensor& src2, const Tensor& src3) override;
  virtual THTensor& addcdiv(const Tensor& src1, scalar_type value,
                            const Tensor& src2, const Tensor& src3) override;
  virtual THTensor& addmv(scalar_type beta, const Tensor& src,
                          scalar_type alpha, const Tensor& mat,
                          const Tensor& vec) override;
  virtual THTensor& addmm(scalar_type beta, const Tensor& src,
                          scalar_type alpha, const Tensor& mat1,
                          const Tensor& mat2) override;
  virtual THTensor& addr(scalar_type beta, const Tensor& src,
                         scalar_type alpha, const Tensor& vec1,
                         const Tensor& vec2) override;
  virtual THTensor& addbmm(scalar_type beta, const Tensor& src,
                           scalar_type alpha, const Tensor& batch1,
                           const Tensor& batch2) override;
  virtual THTensor& baddbmm(scalar_type beta, const Tensor& src,
                            scalar_type alpha, const Tensor& batch1,
                            const Tensor& batch2) override;
  virtual THTensor& match(const Tensor& m1, const Tensor& m2,
                          scalar_type gain) override;
  virtual THTensor& max(const Tensor& indices_, const Tensor& src,
                        int dimension) override;
  virtual THTensor& min(const Tensor& indices_, const Tensor& src,
                        int dimension) override;
  virtual THTensor& kthvalue(const Tensor& indices_, const Tensor& src,
                             long k, int dimension) override;
  virtual THTensor& mode(const Tensor& indices_, const Tensor& src,
                         int dimension) override;
  virtual THTensor& median(const Tensor& indices_, const Tensor& src,
                           int dimension) override;
  virtual THTensor& sum(const Tensor& src, int dimension) override;
  virtual THTensor& prod(const Tensor& src, int dimension) override;
  virtual THTensor& cumsum(const Tensor& src, int dimension) override;
  virtual THTensor& cumprod(const Tensor& src, int dimension) override;
  virtual THTensor& sign(const Tensor& source) override;
  virtual scalar_type trace() override;
  virtual THTensor& cross(const Tensor& src1, const Tensor& src2,
                          int dimension) override;
  virtual THTensor& cmax(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cmin(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cmaxValue(const Tensor& src, scalar_type value) override;
  virtual THTensor& cminValue(const Tensor& src, scalar_type value) override;
  virtual THTensor& zero() override;

  virtual thpp::Type type() const override;
  virtual bool isCuda() const override;
  virtual bool isSparse() const override;
  virtual int getDevice() const override;
  virtual std::unique_ptr<Tensor> newTensor() const override;

private:
  template<typename iterator>
  THTensor& resize(const iterator& begin, const iterator& end);
  template<typename iterator>
  THTensor& resize(const iterator& size_begin, const iterator& size_end,
                   const iterator& stride_begin, const iterator& stride_end);

public:
  tensor_type *tensor;
};

}

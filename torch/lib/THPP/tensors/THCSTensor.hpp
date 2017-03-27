#pragma once

#include <THCS/THCS.h>

// We're defining THCSTensor as a custom class
#undef THCSTensor
#define THCSRealTensor TH_CONCAT_3(THCS,Real,Tensor)

#include "../Tensor.hpp"
#include "../Traits.hpp"

namespace thpp {

template<typename real>
struct thcs_tensor_traits {};

#include "tensors/generic/THCSTensor.hpp"
#include <THCS/THCSGenerateAllTypes.h>

} // namespace thpp

namespace thpp {

template<typename real>
struct THCSTensor : public interface_traits<real>::tensor_interface_type {
  template<typename U>
  friend struct THCSTensor;

private:
  using interface_type = typename interface_traits<real>::tensor_interface_type;
public:
  using tensor_type = typename thcs_tensor_traits<real>::tensor_type;
  using scalar_type = typename interface_type::scalar_type;
  using long_range = Tensor::long_range;

  THCSTensor(THCState* state);
  THCSTensor(THCState* state, tensor_type *wrapped);
  virtual ~THCSTensor();

  virtual THCSTensor* clone() const override;
  virtual THCSTensor* clone_shallow() override;
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
  virtual THCSTensor& retain() override;
  virtual THCSTensor& free() override;

  virtual THCSTensor& resize(const std::initializer_list<long>& new_size) override;
  virtual THCSTensor& resize(const std::vector<long>& new_size) override;
  virtual THCSTensor& resize(THLongStorage *size,
                            THLongStorage *stride) override;
  virtual THCSTensor& resizeAs(const Tensor& src) override;
  virtual THCSTensor& set(const Tensor& src) override;
  virtual THCSTensor& setStorage(const Storage& storage,
                                ptrdiff_t storageOffset,
                                const long_range& size,
                                const long_range& stride) override;
  virtual THCSTensor& setStorage(const Storage& storage,
                                ptrdiff_t storageOffset,
                                THLongStorage *size,
                                THLongStorage *stride) override;

  virtual THCSTensor& narrow(const Tensor& src, int dimension,
                           long firstIndex, long size) override;
  virtual THCSTensor& select(const Tensor& src, int dimension,
                            long sliceIndex) override;
  virtual THCSTensor& transpose(const Tensor& src, int dimension1,
                               int dimension2) override;
  virtual THCSTensor& unfold(const Tensor& src, int dimension,
                            long size, long step) override;
  virtual THCSTensor& squeeze(const Tensor& src, int dimension) override;
  virtual THCSTensor& unsqueeze(const Tensor& src, int dimension) override;

  virtual THCSTensor& diag(const Tensor& src, int k) override;
  virtual THCSTensor& eye(long n, long m) override;
  virtual THCSTensor& range(scalar_type xmin, scalar_type xmax,
                          scalar_type step) override;
  virtual THCSTensor& sort(const Tensor& ri, const Tensor& src,
                       int dimension, int desc) override;
  virtual THCSTensor& topk(const Tensor& ri, const Tensor& src,
                       long k, int dim, int dir, int sorted) override;
  virtual THCSTensor& tril(const Tensor& src, long k) override;
  virtual THCSTensor& triu(const Tensor& src, long k) override;
  // TODO: remove in favor of cat
  virtual THCSTensor& catArray(const std::vector<Tensor*>& inputs,
                             int dimension) override;
  virtual int equal(const Tensor& other) const override;

  // Note: the order in *Value and *Tensor is reversed compared to
  // the declarations in TH/generic/THCSTensorMath.h, so for instance
  // the call THRealTensor_ltTensor(r, ta, tb) is equivalent to
  // ta->ltTensor(r, tb). It is done this way so that the first
  // argument can be casted onto a byte tensor type
  virtual THCSTensor& ltTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCSTensor& leTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCSTensor& gtTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCSTensor& geTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCSTensor& neTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCSTensor& eqTensor(const Tensor& r, const Tensor& tb) override;
  virtual THCSTensor& ltTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THCSTensor& leTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THCSTensor& gtTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THCSTensor& geTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THCSTensor& neTensorT(const Tensor& ta, const Tensor& tb) override;
  virtual THCSTensor& eqTensorT(const Tensor& ta, const Tensor& tb) override;

  virtual THCSTensor& abs(const Tensor& src) override;
  virtual THCSTensor& sigmoid(const Tensor& src) override;
  virtual THCSTensor& log(const Tensor& src) override;
  virtual THCSTensor& log1p(const Tensor& src) override;
  virtual THCSTensor& exp(const Tensor& src) override;
  virtual THCSTensor& cos(const Tensor& src) override;
  virtual THCSTensor& acos(const Tensor& src) override;
  virtual THCSTensor& cosh(const Tensor& src) override;
  virtual THCSTensor& sin(const Tensor& src) override;
  virtual THCSTensor& asin(const Tensor& src) override;
  virtual THCSTensor& sinh(const Tensor& src) override;

  virtual THCSTensor& ltValue(const Tensor& t, scalar_type value) override;
  virtual THCSTensor& leValue(const Tensor& t, scalar_type value) override;
  virtual THCSTensor& gtValue(const Tensor& t, scalar_type value) override;
  virtual THCSTensor& geValue(const Tensor& t, scalar_type value) override;
  virtual THCSTensor& neValue(const Tensor& t, scalar_type value) override;
  virtual THCSTensor& eqValue(const Tensor& t, scalar_type value) override;
  virtual THCSTensor& ltValueT(const Tensor& t, scalar_type value) override;
  virtual THCSTensor& leValueT(const Tensor& t, scalar_type value) override;
  virtual THCSTensor& gtValueT(const Tensor& t, scalar_type value) override;
  virtual THCSTensor& geValueT(const Tensor& t, scalar_type value) override;
  virtual THCSTensor& neValueT(const Tensor& t, scalar_type value) override;
  virtual THCSTensor& eqValueT(const Tensor& t, scalar_type value) override;

  virtual THCSTensor& fill(scalar_type value) override;

  virtual THCSTensor& copy(const Tensor& src) override;
  virtual THCSTensor& cat(const std::vector<Tensor*>& src, int dimension) override;
  virtual THCSTensor& gather(const Tensor& src, int dimension, const Tensor& index) override;
  virtual THCSTensor& scatter(int dimension, const Tensor& index, const Tensor& src) override;
  virtual THCSTensor& scatterFill(int dimension, const Tensor& index, scalar_type value) override;
  virtual scalar_type dot(const Tensor& source) override;
  virtual scalar_type minall() override;
  virtual scalar_type maxall() override;
  virtual scalar_type sumall() override;
  virtual scalar_type prodall() override;
  virtual THCSTensor& neg(const Tensor& src) override;
  virtual THCSTensor& cinv(const Tensor& src) override;
  virtual THCSTensor& add(const Tensor& src, scalar_type value) override;
  virtual THCSTensor& sub(const Tensor& src, scalar_type value) override;
  virtual THCSTensor& mul(const Tensor& src, scalar_type value) override;
  virtual THCSTensor& div(const Tensor& src, scalar_type value) override;
  virtual THCSTensor& fmod(const Tensor& src, scalar_type value) override;
  virtual THCSTensor& remainder(const Tensor& src, scalar_type value) override;
  virtual THCSTensor& clamp(const Tensor& src, scalar_type min_value, scalar_type max_value) override;
  virtual THCSTensor& cadd(const Tensor& src1, const Tensor& src2) override;
  virtual THCSTensor& cadd(const Tensor& src1, scalar_type value, const Tensor& src2) override;
  virtual THCSTensor& csub(const Tensor& src1, scalar_type value, const Tensor& src2) override;
  virtual THCSTensor& cmul(const Tensor& src1, const Tensor& src2) override;
  virtual THCSTensor& cpow(const Tensor& src1, const Tensor& src2) override;
  virtual THCSTensor& cdiv(const Tensor& src1, const Tensor& src2) override;
  virtual THCSTensor& cfmod(const Tensor& src1, const Tensor& src2) override;
  virtual THCSTensor& cremainder(const Tensor& src1, const Tensor& src2) override;
  virtual THCSTensor& addcmul(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) override;
  virtual THCSTensor& addcdiv(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) override;
  virtual THCSTensor& addmv(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat, const Tensor& vec) override;
  virtual THCSTensor& addmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat1, const Tensor& mat2) override;
  virtual THCSTensor& addr(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& vec1, const Tensor& vec2) override;
  virtual THCSTensor& addbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) override;
  virtual THCSTensor& baddbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) override;
  virtual THCSTensor& match(const Tensor& m1, const Tensor& m2, scalar_type gain) override;
  virtual THCSTensor& max(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THCSTensor& min(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THCSTensor& kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension) override;
  virtual THCSTensor& mode(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THCSTensor& median(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THCSTensor& sum(const Tensor& src, int dimension) override;
  virtual THCSTensor& prod(const Tensor& src, int dimension) override;
  virtual THCSTensor& cumsum(const Tensor& src, int dimension) override;
  virtual THCSTensor& cumprod(const Tensor& src, int dimension) override;
  virtual THCSTensor& sign(const Tensor& source) override;
  virtual scalar_type trace() override;
  virtual THCSTensor& cross(const Tensor& src1, const Tensor& src2, int dimension) override;
  virtual THCSTensor& cmax(const Tensor& src1, const Tensor& src2) override;
  virtual THCSTensor& cmin(const Tensor& src1, const Tensor& src2) override;
  virtual THCSTensor& cmaxValue(const Tensor& src, scalar_type value) override;
  virtual THCSTensor& cminValue(const Tensor& src, scalar_type value) override;
  virtual THCSTensor& zero() override;

  virtual thpp::Type type() const override;
  virtual bool isCuda() const override;
  virtual bool isSparse() const override;
  virtual int getDevice() const override;
  virtual std::unique_ptr<Tensor> newTensor() const override;

private:
  template<typename iterator>
  THCSTensor& resize(const iterator& begin, const iterator& end);
  template<typename iterator>
  THCSTensor& resize(const iterator& size_begin, const iterator& size_end,
                    const iterator& stride_begin, const iterator& stride_end);

public:
  tensor_type *tensor;
  THCState *state;
};

}

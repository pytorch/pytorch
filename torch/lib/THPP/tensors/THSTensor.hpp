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
  friend class THSTensor<unsigned char>;
  friend class THSTensor<char>;
  friend class THSTensor<short>;
  friend class THSTensor<int>;
  friend class THSTensor<long>;
  friend class THSTensor<float>;
  friend class THSTensor<double>;

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
  virtual THSTensor& squeeze(const Tensor& src, int dimension) override;
  virtual THSTensor& unsqueeze(const Tensor& src, int dimension) override;

  virtual THSTensor& diag(const Tensor& src, int k) override;
  virtual THSTensor& eye(long n, long m) override;
  virtual THSTensor& range(scalar_type xmin, scalar_type xmax,
                          scalar_type step) override;
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
  virtual THSTensor& max(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THSTensor& min(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THSTensor& kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension) override;
  virtual THSTensor& mode(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THSTensor& median(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THSTensor& sum(const Tensor& src, int dimension) override;
  virtual THSTensor& prod(const Tensor& src, int dimension) override;
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

public:
  tensor_type *tensor;
};

}

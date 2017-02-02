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
  friend class THCTensor<unsigned char>;
  friend class THCTensor<char>;
  friend class THCTensor<short>;
  friend class THCTensor<int>;
  friend class THCTensor<long>;
  friend class THCTensor<float>;
  friend class THCTensor<double>;
  friend class THCTensor<half>;

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
  virtual THCTensor& squeeze(const Tensor& src, int dimension) override;
  virtual THCTensor& unsqueeze(const Tensor& src, int dimension) override;

  virtual THCTensor& fill(scalar_type value) override;

  virtual THCTensor& cat(const std::vector<Tensor*>& src, int dimension) override;
  virtual THCTensor& gather(const Tensor& src, int dimension, const Tensor& index) override;
  virtual THCTensor& scatter(int dimension, const Tensor& index, const Tensor& src) override;
  virtual THCTensor& scatterFill(int dimension, const Tensor& index, scalar_type value) override;
  virtual scalar_type dot(const Tensor& source) override;
  virtual scalar_type minall() override;
  virtual scalar_type maxall() override;
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
  virtual THCTensor& clamp(const Tensor& src, scalar_type min_value, scalar_type max_value) override;
  virtual THCTensor& cadd(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cadd(const Tensor& src1, scalar_type value, const Tensor& src2) override;
  virtual THCTensor& csub(const Tensor& src1, scalar_type value, const Tensor& src2) override;
  virtual THCTensor& cmul(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cpow(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cdiv(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cfmod(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& cremainder(const Tensor& src1, const Tensor& src2) override;
  virtual THCTensor& addcmul(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) override;
  virtual THCTensor& addcdiv(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) override;
  virtual THCTensor& addmv(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat, const Tensor& vec) override;
  virtual THCTensor& addmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat1, const Tensor& mat2) override;
  virtual THCTensor& addr(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& vec1, const Tensor& vec2) override;
  virtual THCTensor& addbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) override;
  virtual THCTensor& baddbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) override;
  virtual THCTensor& match(const Tensor& m1, const Tensor& m2, scalar_type gain) override;
  virtual THCTensor& max(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THCTensor& min(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THCTensor& kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension) override;
  virtual THCTensor& mode(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THCTensor& median(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THCTensor& sum(const Tensor& src, int dimension) override;
  virtual THCTensor& prod(const Tensor& src, int dimension) override;
  virtual THCTensor& cumsum(const Tensor& src, int dimension) override;
  virtual THCTensor& cumprod(const Tensor& src, int dimension) override;
  virtual THCTensor& sign(const Tensor& source) override;
  virtual scalar_type trace() override;
  virtual THCTensor& cross(const Tensor& src1, const Tensor& src2, int dimension) override;
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

public:
  tensor_type *tensor;
  THCState *state;
};

}

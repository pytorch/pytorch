#pragma once

#include "Storage.hpp"
#include "Type.hpp"

#include <TH/TH.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace thpp {

struct Tensor {
  using long_range = std::vector<long>;

  Tensor() {};
  Tensor(const Tensor& other) = delete;
  Tensor(Tensor&& other) = delete;
  virtual ~Tensor() {};

  virtual Tensor* clone() const = 0;
  virtual Tensor* clone_shallow() = 0;
  virtual std::unique_ptr<Tensor> contiguous() const = 0;

  virtual int nDim() const = 0;
  virtual long_range sizes() const = 0;
  virtual long_range strides() const = 0;
  virtual const long* rawSizes() const = 0;
  virtual const long* rawStrides() const = 0;
  virtual std::size_t storageOffset() const = 0;
  virtual std::size_t elementSize() const = 0;
  virtual long long numel() const = 0;
  virtual bool isContiguous() const = 0;
  virtual void* data() = 0;
  virtual const void* data() const = 0;
  virtual void* cdata() = 0;
  virtual const void* cdata() const = 0;
  virtual Tensor& retain() = 0;
  virtual Tensor& free() = 0;

  virtual Tensor& resize(const std::initializer_list<long>& new_size) = 0;
  virtual Tensor& resize(const std::vector<long>& new_size) = 0;
  virtual Tensor& resize(THLongStorage *size,
                         THLongStorage *stride) = 0;
  virtual Tensor& resizeAs(const Tensor& src) = 0;
  virtual Tensor& set(const Tensor& src) = 0;
  virtual Tensor& setStorage(const Storage& storage,
                             ptrdiff_t storageOffset,
                             const long_range& size,
                             const long_range& stride) = 0;
  virtual Tensor& setStorage(const Storage& storage,
                             ptrdiff_t storageOffset,
                             THLongStorage *size,
                             THLongStorage *stride) = 0;
  virtual Tensor& narrow(const Tensor& src,
                         int dimension,
                         long firstIndex,
                         long size) = 0;
  virtual Tensor& select(const Tensor& src, int dimension, long sliceIndex) = 0;
  virtual Tensor& transpose(const Tensor& src, int dimension1, int dimension2) = 0;
  virtual Tensor& unfold(const Tensor& src, int dimension, long size, long step) = 0;
  virtual Tensor& squeeze(const Tensor& src, int dimension) = 0;
  virtual Tensor& unsqueeze(const Tensor& src, int dimension) = 0;

  virtual Tensor& copy(const Tensor& src) = 0;
  virtual Tensor& cat(const std::vector<Tensor*>& src, int dimension) = 0;
  virtual Tensor& gather(const Tensor& src, int dimension, const Tensor& index) = 0;
  virtual Tensor& scatter(int dimension, const Tensor& index, const Tensor& src) = 0;
  virtual Tensor& neg(const Tensor& src) = 0;
  virtual Tensor& cadd(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& cinv(const Tensor& src) = 0;
  virtual Tensor& cmul(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& cpow(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& cdiv(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& cfmod(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& cremainder(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& max(const Tensor& indices_, const Tensor& src, int dimension) = 0;
  virtual Tensor& min(const Tensor& indices_, const Tensor& src, int dimension) = 0;
  virtual Tensor& kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension) = 0;
  virtual Tensor& mode(const Tensor& indices_, const Tensor& src, int dimension) = 0;
  virtual Tensor& median(const Tensor& indices_, const Tensor& src, int dimension) = 0;
  virtual Tensor& sum(const Tensor& src, int dimension) = 0;
  virtual Tensor& prod(const Tensor& src, int dimension) = 0;
  virtual Tensor& cumsum(const Tensor& src, int dimension) = 0;
  virtual Tensor& cumprod(const Tensor& src, int dimension) = 0;
  virtual Tensor& sign(const Tensor& source) = 0;
  virtual Tensor& cross(const Tensor& src1, const Tensor& src2, int dimension) = 0;
  virtual Tensor& cmax(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& cmin(const Tensor& src1, const Tensor& src2) = 0;
  virtual Tensor& zero() = 0;

  virtual Tensor& diag(const Tensor& src, int k) = 0;
  virtual Tensor& eye(long n, long m) = 0;
  // virtual Tensor& randperm() = 0; TODO
  virtual Tensor& sort(const Tensor& ri, const Tensor& src,
                       int dimension, int desc) = 0;
  virtual Tensor& topk(const Tensor& ri, const Tensor& src,
                       long k, int dim, int dir, int sorted) = 0;
  virtual Tensor& tril(const Tensor& src, long k) = 0;
  virtual Tensor& triu(const Tensor& src, long k) = 0;
  virtual Tensor& catArray(const std::vector<Tensor*>& inputs, int dimension) = 0;
  virtual int equal(const Tensor& other) const = 0;
  virtual Tensor& ltTensor(const Tensor& r, const Tensor& tb) = 0;
  virtual Tensor& leTensor(const Tensor& r, const Tensor& tb) = 0;
  virtual Tensor& gtTensor(const Tensor& r, const Tensor& tb) = 0;
  virtual Tensor& geTensor(const Tensor& r, const Tensor& tb) = 0;
  virtual Tensor& neTensor(const Tensor& r, const Tensor& tb) = 0;
  virtual Tensor& eqTensor(const Tensor& r, const Tensor& tb) = 0;
  virtual Tensor& ltTensorT(const Tensor& ta, const Tensor& tb) = 0;
  virtual Tensor& leTensorT(const Tensor& ta, const Tensor& tb) = 0;
  virtual Tensor& gtTensorT(const Tensor& ta, const Tensor& tb) = 0;
  virtual Tensor& geTensorT(const Tensor& ta, const Tensor& tb) = 0;
  virtual Tensor& neTensorT(const Tensor& ta, const Tensor& tb) = 0;
  virtual Tensor& eqTensorT(const Tensor& ta, const Tensor& tb) = 0;
  virtual Tensor& abs(const Tensor& src) = 0;
  virtual Tensor& sigmoid(const Tensor& src) = 0;
  virtual Tensor& log(const Tensor& src) = 0;
  virtual Tensor& log1p(const Tensor& src) = 0;
  virtual Tensor& exp(const Tensor& src) = 0;
  virtual Tensor& cos(const Tensor& src) = 0;
  virtual Tensor& acos(const Tensor& src) = 0;
  virtual Tensor& cosh(const Tensor& src) = 0;
  virtual Tensor& sin(const Tensor& src) = 0;
  virtual Tensor& asin(const Tensor& src) = 0;
  virtual Tensor& sinh(const Tensor& src) = 0;

  virtual thpp::Type type() const = 0;
  virtual bool isCuda() const = 0;
  virtual bool isSparse() const = 0;
  virtual int getDevice() const = 0;
  virtual std::unique_ptr<Tensor> newTensor() const = 0;
};

template<typename real>
struct TensorScalarInterface : public Tensor {
  using Tensor::cadd;
  using scalar_type = real;
  virtual TensorScalarInterface& fill(scalar_type value) = 0;

  virtual TensorScalarInterface& scatterFill(int dimension, const Tensor& index, scalar_type value) = 0;
  virtual scalar_type dot(const Tensor& source) = 0;
  virtual scalar_type minall() = 0;
  virtual scalar_type maxall() = 0;
  virtual scalar_type sumall() = 0;
  virtual scalar_type prodall() = 0;
  virtual TensorScalarInterface& add(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& sub(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& mul(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& div(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& fmod(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& remainder(const Tensor& src, scalar_type value) = 0;
  virtual TensorScalarInterface& clamp(const Tensor& src, scalar_type min_value,
                                       scalar_type max_value) = 0;
  virtual TensorScalarInterface& cadd(const Tensor& src1, scalar_type value,
                                      const Tensor& src2) = 0;
  virtual TensorScalarInterface& csub(const Tensor& src1, scalar_type value,
                                      const Tensor& src2) = 0;
  virtual TensorScalarInterface& addcmul(const Tensor& src1, scalar_type value,
                                         const Tensor& src2,
                                         const Tensor& src3) = 0;
  virtual TensorScalarInterface& addcdiv(const Tensor& src1, scalar_type value,
                                         const Tensor& src2,
                                         const Tensor& src3) = 0;
  virtual TensorScalarInterface& addmv(scalar_type beta, const Tensor& src,
                                       scalar_type alpha, const Tensor& mat,
                                       const Tensor& vec) = 0;
  virtual TensorScalarInterface& addmm(scalar_type beta, const Tensor& src,
                                       scalar_type alpha, const Tensor& mat1,
                                       const Tensor& mat2) = 0;
  virtual TensorScalarInterface& addr(scalar_type beta, const Tensor& src,
                                      scalar_type alpha, const Tensor& vec1,
                                      const Tensor& vec2) = 0;
  virtual TensorScalarInterface& addbmm(scalar_type beta, const Tensor& src,
                                        scalar_type alpha, const Tensor& batch1,
                                        const Tensor& batch2) = 0;
  virtual TensorScalarInterface& baddbmm(scalar_type beta, const Tensor& src,
                                         scalar_type alpha, const Tensor& batch1,
                                         const Tensor& batch2) = 0;
  virtual TensorScalarInterface& match(const Tensor& m1, const Tensor& m2,
                                       scalar_type gain) = 0;
  virtual scalar_type trace() = 0;
  virtual TensorScalarInterface& cmaxValue(const Tensor& src,
                                           scalar_type value) = 0;
  virtual TensorScalarInterface& cminValue(const Tensor& src,
                                           scalar_type value) = 0;

  virtual TensorScalarInterface& range(scalar_type xmin, scalar_type xmax,
                                       scalar_type step) = 0;
  virtual TensorScalarInterface& ltValue(const Tensor& t, scalar_type value) = 0;
  virtual TensorScalarInterface& leValue(const Tensor& t, scalar_type value) = 0;
  virtual TensorScalarInterface& gtValue(const Tensor& t, scalar_type value) = 0;
  virtual TensorScalarInterface& geValue(const Tensor& t, scalar_type value) = 0;
  virtual TensorScalarInterface& neValue(const Tensor& t, scalar_type value) = 0;
  virtual TensorScalarInterface& eqValue(const Tensor& t, scalar_type value) = 0;
  virtual TensorScalarInterface& ltValueT(const Tensor& t, scalar_type value) = 0;
  virtual TensorScalarInterface& leValueT(const Tensor& t, scalar_type value) = 0;
  virtual TensorScalarInterface& gtValueT(const Tensor& t, scalar_type value) = 0;
  virtual TensorScalarInterface& geValueT(const Tensor& t, scalar_type value) = 0;
  virtual TensorScalarInterface& neValueT(const Tensor& t, scalar_type value) = 0;
  virtual TensorScalarInterface& eqValueT(const Tensor& t, scalar_type value) = 0;
};

using FloatTensor = TensorScalarInterface<double>;
using IntTensor = TensorScalarInterface<long long>;

} // namespace thpp

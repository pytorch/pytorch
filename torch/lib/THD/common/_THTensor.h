#pragma once

#include <TH/TH.h>

#undef THTensor
#define THRealTensor TH_CONCAT_3(TH,Real,Tensor)

template<typename real>
struct th_traits {};

#include "generic/_THTensor.h"
#include <TH/THGenerateAllTypes.h>

template<typename real>
struct THTensor : public tensor_traits<real>::interface_type {
private:
  using interface_type = typename tensor_traits<real>::interface_type;
public:
  using tensor_type = typename th_traits<real>::tensor_type;
  using scalar_type = typename interface_type::scalar_type;
  using long_range = Tensor::long_range;

  THTensor();
  THTensor(tensor_type *wrapped);
  virtual ~THTensor();

  virtual int nDim() const override;
  virtual long_range sizes() const override;
  virtual long_range strides() const override;
  virtual const long* rawSizes() const override;
  virtual const long* rawStrides() const override;
  virtual size_t storageOffset() const override;
  virtual size_t elementSize() const override;
  virtual void* data() override;
  virtual const void* data() const override;

  virtual THTensor& resize(const std::initializer_list<long> &new_size) override;

  virtual THTensor& fill(scalar_type value) override;
  virtual THTensor& add(const Tensor &source, scalar_type scalar) override;

protected:
  tensor_type *tensor;
};

#pragma once

#include "ATen/Tensor.h"
#include "ATen/dlpack.h"

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor object
// 2) take a dlpack tensor and convert it to the Tensor object

namespace at { namespace dlpack {

// create the shared pointers typedef
using DLTensorSPtr = std::shared_ptr<DLTensor>;

class DLConvertor {
  public:
    // constructor for the Tensor types, can be null pointers
    // DLConvertor();
    explicit DLConvertor(Tensor& atTensor)
      : atTensor_(atTensor) {}

    // TODO: what is the proper way to destruct this?
    ~DLConvertor() = default;

    DLDataType getDLDataType(const Type& type);
    DLContext getDLContext(const Type& type, const int64_t& device_id);
    int64_t* getDLInt64Array(const IntList& arr);
    DLTensorSPtr convertToDLTensor(const Tensor& atTensor);

  private:
    // pass the pointers to the dlTensor or the aTensor and get the conversions
    Tensor atTensor_;
};

}} //namespace at::dlpack

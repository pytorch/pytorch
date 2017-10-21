#pragma once

#include "ATen/Tensor.h"
#include "ATen/ATen.h"
#include "ATen/dlpack.h"

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor
// 2) take a dlpack tensor and convert it to the ATen Tensor

namespace at {

ATen_CLASS ScalarType toScalarType(const DLDataType& dtype);
ATen_CLASS DLManagedTensor * toDLPack(const Tensor& src);
ATen_CLASS Tensor fromDLPack(const DLManagedTensor* src);

} //namespace at

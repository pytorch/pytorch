#pragma once

#include "ATen/Tensor.h"
#include "ATen/ATen.h"
#include "ATen/dlpack.h"

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor
// 2) take a dlpack tensor and convert it to the ATen Tensor

namespace at {


struct ATenDLMTensor {
  Tensor handle;
  DLManagedTensor tensor;
};

DLManagedTensor * toDLPack(const Tensor& src);
Tensor fromDLPack(const DLManagedTensor* src);

} //namespace at

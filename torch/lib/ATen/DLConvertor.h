#pragma once

#include "ATen/Tensor.h"
#include "ATen/ATen.h"
#include "ATen/dlpack.h"

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor object
// 2) take a dlpack tensor and convert it to the Tensor object

namespace at {

DLTensor* toDLPack(const Tensor& src, DLTensor* dlTensor);
Tensor fromDLPack(const DLTensor* src);

} //namespace at

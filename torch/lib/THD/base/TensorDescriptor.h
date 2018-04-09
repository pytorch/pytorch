#pragma once

#include "../THD.h"
#include <TH/TH.h>
#ifdef WITH_CUDA
#include <THC/THC.h>
#endif

#ifndef _THD_CORE
#include <ATen/ATen.h>
using THDTensorDescriptor = at::Tensor;
#endif

THDTensorDescriptor THDTensorDescriptor_newFromTHDoubleTensor(THDoubleTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHFloatTensor(THFloatTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHLongTensor(THLongTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHIntTensor(THIntTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHShortTensor(THShortTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHCharTensor(THCharTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHByteTensor(THByteTensor *tensor);
#ifdef WITH_CUDA
THDTensorDescriptor THDTensorDescriptor_newFromTHCudaDoubleTensor(THCudaDoubleTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHCudaFloatTensor(THCudaTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHCudaHalfTensor(THCudaHalfTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHCudaLongTensor(THCudaLongTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHCudaIntTensor(THCudaIntTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHCudaShortTensor(THCudaShortTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHCudaCharTensor(THCudaCharTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHCudaByteTensor(THCudaByteTensor *tensor);
#endif

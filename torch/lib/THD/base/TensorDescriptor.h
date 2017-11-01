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

THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHDoubleTensor(THDoubleTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHFloatTensor(THFloatTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHLongTensor(THLongTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHIntTensor(THIntTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHShortTensor(THShortTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHCharTensor(THCharTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHByteTensor(THByteTensor *tensor);
#ifdef WITH_CUDA
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHCudaDoubleTensor(THCudaDoubleTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHCudaFloatTensor(THCudaTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHCudaLongTensor(THCudaLongTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHCudaIntTensor(THCudaIntTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHCudaShortTensor(THCudaShortTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHCudaCharTensor(THCudaCharTensor *tensor);
THD_API THDTensorDescriptor THDTensorDescriptor_newFromTHCudaByteTensor(THCudaByteTensor *tensor);
#endif

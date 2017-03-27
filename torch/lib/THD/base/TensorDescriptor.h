#pragma once

#include "../THD.h"
#include <TH/TH.h>

#ifndef _THD_CORE
struct _THDTensorDescriptor;
typedef struct _THDTensorDescriptor THDTensorDescriptor;
#endif

THD_API THDTensorDescriptor* THDTensorDescriptor_newFromTHDoubleTensor(THDoubleTensor *tensor);
THD_API THDTensorDescriptor* THDTensorDescriptor_newFromTHFloatTensor(THFloatTensor *tensor);
THD_API THDTensorDescriptor* THDTensorDescriptor_newFromTHLongTensor(THLongTensor *tensor);
THD_API THDTensorDescriptor* THDTensorDescriptor_newFromTHIntTensor(THIntTensor *tensor);
THD_API THDTensorDescriptor* THDTensorDescriptor_newFromTHShortTensor(THShortTensor *tensor);
THD_API THDTensorDescriptor* THDTensorDescriptor_newFromTHCharTensor(THCharTensor *tensor);
THD_API THDTensorDescriptor* THDTensorDescriptor_newFromTHByteTensor(THByteTensor *tensor);
THD_API void THDTensorDescriptor_free(THDTensorDescriptor* desc);

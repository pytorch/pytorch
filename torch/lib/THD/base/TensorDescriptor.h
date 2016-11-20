#pragma once

#include <TH/TH.h>

#ifndef _THD_CORE
struct _THDTensorDescriptor;
typedef struct _THDTensorDescriptor* THDTensorDescriptor;
#endif

THDTensorDescriptor THDTensorDescriptor_newFromTHDoubleTensor(THDoubleTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHFloatTensor(THFloatTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHLongTensor(THFloatTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHIntTensor(THFloatTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHShortTensor(THFloatTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHCharTensor(THFloatTensor *tensor);
THDTensorDescriptor THDTensorDescriptor_newFromTHByteTensor(THFloatTensor *tensor);

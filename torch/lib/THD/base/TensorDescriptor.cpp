#include "TensorDescriptor.hpp"


THDTensorDescriptor THDTensorDescriptor_newFromTHDoubleTensor(THDoubleTensor *tensor) {
  return new THTensor<double>(tensor);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHFloatTensor(THFloatTensor *tensor) {
  return new THTensor<float>(tensor);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHLongTensor(THLongTensor *tensor) {
  return new THTensor<long>(tensor);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHIntTensor(THIntTensor *tensor) {
  return new THTensor<int>(tensor);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHShortTensor(THShortTensor *tensor) {
  return new THTensor<short>(tensor);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHCharTensor(THCharTensor *tensor) {
  return new THTensor<char>(tensor);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHByteTensor(THByteTensor *tensor) {
  return new THTensor<unsigned char>(tensor);
}

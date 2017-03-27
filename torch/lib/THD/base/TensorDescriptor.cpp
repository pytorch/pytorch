#include "TensorDescriptor.hpp"
#include <THPP/tensors/THTensor.hpp>

THDTensorDescriptor* THDTensorDescriptor_newFromTHDoubleTensor(THDoubleTensor *tensor) {
  THDoubleTensor_retain(tensor);
  return new thpp::THTensor<double>(tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHFloatTensor(THFloatTensor *tensor) {
  THFloatTensor_retain(tensor);
  return new thpp::THTensor<float>(tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHLongTensor(THLongTensor *tensor) {
  THLongTensor_retain(tensor);
  return new thpp::THTensor<long>(tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHIntTensor(THIntTensor *tensor) {
  THIntTensor_retain(tensor);
  return new thpp::THTensor<int>(tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHShortTensor(THShortTensor *tensor) {
  THShortTensor_retain(tensor);
  return new thpp::THTensor<short>(tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHCharTensor(THCharTensor *tensor) {
  THCharTensor_retain(tensor);
  return new thpp::THTensor<char>(tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHByteTensor(THByteTensor *tensor) {
  THByteTensor_retain(tensor);
  return new thpp::THTensor<unsigned char>(tensor);
}

THD_API void THDTensorDescriptor_free(THDTensorDescriptor* desc) {
  delete desc;
}

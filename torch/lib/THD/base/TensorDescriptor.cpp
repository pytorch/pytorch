#include "TensorDescriptor.hpp"
#include <THPP/tensors/THTensor.hpp>
#ifdef WITH_CUDA
#include <THPP/tensors/THCTensor.hpp>
#endif

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

#ifdef WITH_CUDA
extern THCState* state;

THDTensorDescriptor* THDTensorDescriptor_newFromTHCudaDoubleTensor(THCudaDoubleTensor *tensor) {
  THCudaDoubleTensor_retain(state, tensor);
  return new thpp::THCTensor<double>(state, tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHCudaFloatTensor(THCudaTensor *tensor) {
  THCudaTensor_retain(state, tensor);
  return new thpp::THCTensor<float>(state, tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHCudaLongTensor(THCudaLongTensor *tensor) {
  THCudaLongTensor_retain(state, tensor);
  return new thpp::THCTensor<long>(state, tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHCudaIntTensor(THCudaIntTensor *tensor) {
  THCudaIntTensor_retain(state, tensor);
  return new thpp::THCTensor<int>(state, tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHCudaShortTensor(THCudaShortTensor *tensor) {
  THCudaShortTensor_retain(state, tensor);
  return new thpp::THCTensor<short>(state, tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHCudaCharTensor(THCudaCharTensor *tensor) {
  THCudaCharTensor_retain(state, tensor);
  return new thpp::THCTensor<char>(state, tensor);
}

THDTensorDescriptor* THDTensorDescriptor_newFromTHCudaByteTensor(THCudaByteTensor *tensor) {
  THCudaByteTensor_retain(state, tensor);
  return new thpp::THCTensor<unsigned char>(state, tensor);
}
#endif

THD_API void THDTensorDescriptor_free(THDTensorDescriptor* desc) {
  delete desc;
}

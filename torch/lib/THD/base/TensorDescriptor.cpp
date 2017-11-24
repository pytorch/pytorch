#include "TensorDescriptor.hpp"
#include "Cuda.hpp"

THDTensorDescriptor THDTensorDescriptor_newFromTHDoubleTensor(THDoubleTensor *tensor) {
  return at::getType(at::Backend::CPU, at::ScalarType::Double).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHFloatTensor(THFloatTensor *tensor) {
  return at::getType(at::Backend::CPU, at::ScalarType::Float).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHLongTensor(THLongTensor *tensor) {
  return at::getType(at::Backend::CPU, at::ScalarType::Long).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHIntTensor(THIntTensor *tensor) {
  return at::getType(at::Backend::CPU, at::ScalarType::Int).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHShortTensor(THShortTensor *tensor) {
  return at::getType(at::Backend::CPU, at::ScalarType::Short).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHCharTensor(THCharTensor *tensor) {
  return at::getType(at::Backend::CPU, at::ScalarType::Char).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHByteTensor(THByteTensor *tensor) {
  return at::getType(at::Backend::CPU, at::ScalarType::Byte).unsafeTensorFromTH((void*)tensor, true);
}

#ifdef WITH_CUDA

THDTensorDescriptor THDTensorDescriptor_newFromTHCudaDoubleTensor(THCudaDoubleTensor *tensor) {
  return at::getType(at::Backend::CUDA, at::ScalarType::Double).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHCudaFloatTensor(THCudaTensor *tensor) {
  return at::getType(at::Backend::CUDA, at::ScalarType::Float).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHCudaLongTensor(THCudaLongTensor *tensor) {
  return at::getType(at::Backend::CUDA, at::ScalarType::Long).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHCudaIntTensor(THCudaIntTensor *tensor) {
  return at::getType(at::Backend::CUDA, at::ScalarType::Int).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHCudaShortTensor(THCudaShortTensor *tensor) {
  return at::getType(at::Backend::CUDA, at::ScalarType::Short).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHCudaCharTensor(THCudaCharTensor *tensor) {
  return at::getType(at::Backend::CUDA, at::ScalarType::Char).unsafeTensorFromTH((void*)tensor, true);
}

THDTensorDescriptor THDTensorDescriptor_newFromTHCudaByteTensor(THCudaByteTensor *tensor) {
  return at::getType(at::Backend::CUDA, at::ScalarType::Byte).unsafeTensorFromTH((void*)tensor, true);
}
#endif

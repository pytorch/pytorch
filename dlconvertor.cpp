#include "ATen/dlconvertor.h"

#include <iostream>
#include <sstream>

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor object
// 2) take a dlpack tensor and convert it to the Tensor object

using namespace std;
namespace at {
namespace dlpack {

DLDataType DLConvertor::getDLDataType(const Type& type) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = type.elementSizeInBytes() * 8;
  switch (type.scalarType()) {
    case ScalarType::Byte:
      dtype.code = DLDataTypeCode::kUInt;
      break;
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kInt;
      break;
    case ScalarType::Double:
      dtype.code = DLDataTypeCode::kFloat;
      break;
    case ScalarType::Float:
      dtype.code = DLDataTypeCode::kFloat;
      break;
    case ScalarType::Int:
      dtype.code = DLDataTypeCode::kInt;
      break;
    case ScalarType::Long:
      dtype.code = DLDataTypeCode::kInt;
      break;
    case ScalarType::Short:
      dtype.code = DLDataTypeCode::kInt;
      break;
    case ScalarType::Half:
      dtype.code = DLDataTypeCode::kFloat;
      break;
    case ScalarType::NumOptions:
      throw std::logic_error("NumOptions is not a valid ScalarType");
  }
  return dtype;
}


DLContext DLConvertor::getDLContext(
    const Type& type, const int64_t& device_id) {
  DLContext ctx;
  ctx.device_id = device_id;
  if (type.isCuda()) {
    ctx.device_type = DLDeviceType::kGPU;
  } else {
    ctx.device_type = DLDeviceType::kCPU;
  }
  return ctx;
}


int64_t* DLConvertor::getDLInt64Array(const IntList& arr) {
  size_t arrLen = arr.size();
  auto out = new int64_t[arrLen];
  for (size_t i = 0; i < arrLen; i++) {
    out[i] = arr[i];
  }
  return out;
}


// This function returns a shared_ptr to DLpack tensor constructed out ATen tensor
DLTensorSPtr DLConvertor::convertToDLTensor(const Tensor& atTensor) {
  std::cout << "Inside the convertToDLTensor method\n" << std::endl;
  DLTensorSPtr dlTensor(new DLTensor);

  dlTensor->data = atTensor.data_ptr();
  // TODO: get_device() throws error
  // int64_t device_id = atTensor.get_device();
  int64_t device_id = 0;

  dlTensor->ctx = getDLContext(atTensor.type(), device_id);
  dlTensor->ndim = atTensor.dim();
  dlTensor->dtype = getDLDataType(atTensor.type());
  dlTensor->shape = getDLInt64Array(atTensor.sizes());
  dlTensor->strides = getDLInt64Array(atTensor.strides());
  // TODO: what is the correct offset?
  dlTensor->byte_offset = 0;
  return dlTensor;
}

} // namespace dlpack
} //namespace at

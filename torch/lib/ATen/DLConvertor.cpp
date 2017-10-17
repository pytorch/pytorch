#include "ATen/DLConvertor.h"

#include <iostream>
#include <sstream>


using namespace std;
namespace at {

static DLDataType getDLDataType(const Type& type) {
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


static DLContext getDLContext(const Type& type, const int64_t& device_id) {
  DLContext ctx;
  ctx.device_id = device_id;
  if (type.isCuda()) {
    ctx.device_type = DLDeviceType::kGPU;
  } else {
    ctx.device_type = DLDeviceType::kCPU;
  }
  return ctx;
}


static Backend getATenBackend(const DLContext& ctx) {
  Backend backend;
  switch (ctx.device_type) {
    case DLDeviceType::kCPU:
      backend = Backend::CPU;
      break;
    case DLDeviceType::kGPU:
      backend = Backend::CUDA;
      break;
    default:
      throw std::logic_error("Unsupported device_type: " + std::to_string(ctx.device_type));
  }
  return backend;
}


// TODO: use macros?
static ScalarType getATenScalarType(const DLDataType& dtype) {
  ScalarType stype;
  if (dtype.lanes != 1) throw std::logic_error("ATen does not support lanes != 1");
  switch (dtype.code) {
    case DLDataTypeCode::kUInt:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Byte;
          break;
        default:
          throw std::logic_error("Unsupported kUInt bits " + std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kInt:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Char;
          break;
        case 16:
          stype = ScalarType::Short;
          break;
        case 32:
          stype = ScalarType::Int;
          break;
        case 64:
          stype = ScalarType::Long;
          break;
        default:
          throw std::logic_error("Unsupported kInt bits " + std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kFloat:
      switch (dtype.bits) {
        case 16:
          stype = ScalarType::Half;
          break;
        case 32:
          stype = ScalarType::Float;
          break;
        case 64:
          stype = ScalarType::Double;
          break;
        default:
          throw std::logic_error("Unsupported kFloat bits " + std::to_string(dtype.bits));
      }
      break;
    default:
      throw std::logic_error("Unsupported code " + std::to_string(dtype.code));
  }
  return stype;
}


// This function returns a shared_ptr to DLpack tensor constructed out ATen tensor
DLTensor* toDLPack(const Tensor& src, DLTensor* dlTensor) {
  dlTensor->data = src.data_ptr();
  int64_t device_id = 0;
  if (src.type().isCuda()) {
    device_id = src.get_device();
  }
  dlTensor->ctx = getDLContext(src.type(), device_id);
  dlTensor->ndim = src.dim();
  dlTensor->dtype = getDLDataType(src.type());
  dlTensor->shape = const_cast<int64_t*>(src.sizes().data());
  dlTensor->strides = const_cast<int64_t*>(src.strides().data());
  dlTensor->byte_offset = 0;
  return dlTensor;
}


Tensor fromDLPack(const DLTensor* src) {
  Backend backend = getATenBackend(src->ctx);
  ScalarType stype = getATenScalarType(src->dtype);
  return getType(backend, stype).tensorFromBlob(
      src->data,
      IntList(src->shape, src->ndim), IntList(src->strides, src->ndim));
}
} //namespace at

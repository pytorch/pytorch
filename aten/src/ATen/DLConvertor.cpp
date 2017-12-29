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
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Int:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Long:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Undefined:
      throw std::logic_error("Undefined is not a valid ScalarType");
    case ScalarType::NumOptions:
      throw std::logic_error("NumOptions is not a valid ScalarType");
  }
  return dtype;
}


static DLContext getDLContext(const Type& type, const int64_t& device_id) {
  DLContext ctx;
  ctx.device_id = device_id;
  if (type.is_cuda()) {
    ctx.device_type = DLDeviceType::kDLGPU;
  } else {
    ctx.device_type = DLDeviceType::kDLCPU;
  }
  return ctx;
}


static Backend getATenBackend(const DLContext& ctx) {
  Backend backend;
  switch (ctx.device_type) {
    case DLDeviceType::kDLCPU:
      backend = Backend::CPU;
      break;
    case DLDeviceType::kDLGPU:
      backend = Backend::CUDA;
      break;
    default:
      throw std::logic_error("Unsupported device_type: " + std::to_string(ctx.device_type));
  }
  return backend;
}


ScalarType toScalarType(const DLDataType& dtype) {
  ScalarType stype;
  if (dtype.lanes != 1) throw std::logic_error("ATen does not support lanes != 1");
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Byte;
          break;
        default:
          throw std::logic_error("Unsupported kUInt bits " + std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLInt:
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
    case DLDataTypeCode::kDLFloat:
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

struct ATenDLMTensor {
  Tensor handle;
  DLManagedTensor tensor;
};

void deleter(DLManagedTensor * arg) {
  delete static_cast<ATenDLMTensor*>(arg->manager_ctx);
}


// This function returns a shared_ptr to memory managed DLpack tensor constructed
// out of ATen tensor
DLManagedTensor* toDLPack(const Tensor& src) {
  ATenDLMTensor * atDLMTensor(new ATenDLMTensor);
  atDLMTensor->handle = src;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.deleter = &deleter;
  atDLMTensor->tensor.dl_tensor.data = src.data_ptr();
  int64_t device_id = 0;
  if (src.type().is_cuda()) {
    device_id = src.get_device();
  }
  atDLMTensor->tensor.dl_tensor.ctx = getDLContext(src.type(), device_id);
  atDLMTensor->tensor.dl_tensor.ndim = src.dim();
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataType(src.type());
  atDLMTensor->tensor.dl_tensor.shape = const_cast<int64_t*>(src.sizes().data());
  atDLMTensor->tensor.dl_tensor.strides = const_cast<int64_t*>(src.strides().data());
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  return &(atDLMTensor->tensor);
}


Tensor fromDLPack(const DLManagedTensor* src) {
  Backend backend = getATenBackend(src->dl_tensor.ctx);
  ScalarType stype = toScalarType(src->dl_tensor.dtype);
  auto deleter = [src](void * self) {
    src->deleter(const_cast<DLManagedTensor*>(src));
  };
  return getType(backend, stype).tensorFromBlob(
      src->dl_tensor.data,
      IntList(src->dl_tensor.shape, src->dl_tensor.ndim),
      IntList(src->dl_tensor.strides, src->dl_tensor.ndim),
      deleter);
}
} //namespace at

#include <ATen/DLConvertor.h>
#include <memory>

using namespace std;
namespace at {

DLDataType getDLDataType(const Tensor& t) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = t.element_size() * 8;
  switch (t.scalar_type()) {
    case ScalarType::UInt1:
    case ScalarType::UInt2:
    case ScalarType::UInt3:
    case ScalarType::UInt4:
    case ScalarType::UInt5:
    case ScalarType::UInt6:
    case ScalarType::UInt7:
    case ScalarType::Byte:
    case ScalarType::UInt16:
    case ScalarType::UInt32:
    case ScalarType::UInt64:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    case ScalarType::Int1:
    case ScalarType::Int2:
    case ScalarType::Int3:
    case ScalarType::Int4:
    case ScalarType::Int5:
    case ScalarType::Int6:
    case ScalarType::Int7:
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    // NOLINTNEXTLINE(bugprone-branch-clone)
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
    case ScalarType::Bool:
      dtype.code = DLDataTypeCode::kDLBool;
      break;
    case ScalarType::ComplexHalf:
    case ScalarType::ComplexFloat:
    case ScalarType::ComplexDouble:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case ScalarType::BFloat16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      break;
    // TODO(#146647): use macro here instead of spelling out each shell dtype
    case ScalarType::Float8_e5m2:
      dtype.code = DLDataTypeCode::kDLFloat8_e5m2;
      break;
    case ScalarType::Float8_e5m2fnuz:
      dtype.code = DLDataTypeCode::kDLFloat8_e5m2fnuz;
      break;
    case ScalarType::Float8_e4m3fn:
      dtype.code = DLDataTypeCode::kDLFloat8_e4m3fn;
      break;
    case ScalarType::Float8_e4m3fnuz:
      dtype.code = DLDataTypeCode::kDLFloat8_e4m3fnuz;
      break;
    case ScalarType::Float8_e8m0fnu:
      dtype.code = DLDataTypeCode::kDLFloat8_e8m0fnu;
      break;
    case ScalarType::Float4_e2m1fn_x2:
      dtype.code = DLDataTypeCode::kDLFloat4_e2m1fn;
      dtype.lanes = 2;
      dtype.bits = 4;
      break;
    case ScalarType::QInt8:
    case ScalarType::QUInt8:
    case ScalarType::QInt32:
    case ScalarType::QUInt4x2:
    case ScalarType::QUInt2x4:
      TORCH_CHECK_BUFFER(false, "QUInt/QInt types are not supported by dlpack");
      break;
    case ScalarType::Bits1x8:
    case ScalarType::Bits2x4:
    case ScalarType::Bits4x2:
    case ScalarType::Bits8:
    case ScalarType::Bits16:
      TORCH_CHECK_BUFFER(false, "Bit types are not supported by dlpack");
      break;
    case ScalarType::Undefined:
      TORCH_CHECK_BUFFER(false, "Undefined is not a valid ScalarType");
    case ScalarType::NumOptions:
      TORCH_CHECK_BUFFER(false, "NumOptions is not a valid ScalarType");
  }
  return dtype;
}

DLDevice torchDeviceToDLDevice(at::Device device) {
  DLDevice ctx;

  ctx.device_id = (device.is_cuda() || device.is_privateuseone())
      ? static_cast<int32_t>(static_cast<unsigned char>(device.index()))
      : 0;

  switch (device.type()) {
    case DeviceType::CPU:
      ctx.device_type = DLDeviceType::kDLCPU;
      break;
    case DeviceType::CUDA:
#ifdef USE_ROCM
      // ROCM, if enabled will look like cuda to PyTorch
      // while everyone else should see HIP
      ctx.device_type = DLDeviceType::kDLROCM;
#else
      ctx.device_type = DLDeviceType::kDLCUDA;
#endif
      break;
    case DeviceType::OPENCL:
      ctx.device_type = DLDeviceType::kDLOpenCL;
      break;
    case DeviceType::HIP:
      ctx.device_type = DLDeviceType::kDLROCM;
      break;
    case DeviceType::XPU:
      ctx.device_type = DLDeviceType::kDLOneAPI;
      ctx.device_id = at::detail::getXPUHooks().getGlobalIdxFromDevice(device);
      break;
    case DeviceType::MAIA:
      ctx.device_type = DLDeviceType::kDLMAIA;
      break;
    case DeviceType::PrivateUse1:
      ctx.device_type = DLDeviceType::kDLExtDev;
      break;
    case DeviceType::MPS:
      ctx.device_type = DLDeviceType::kDLMetal;
      break;
    default:
      TORCH_CHECK_BUFFER(false, "Cannot pack tensors on " + device.str());
  }

  return ctx;
}

Device dlDeviceToTorchDevice(
    DLDeviceType type,
    c10::DeviceIndex index,
    void* data) {
  switch (type) {
    case DLDeviceType::kDLCPU:
      return at::Device(DeviceType::CPU);
#ifndef USE_ROCM
    // if we are compiled under HIP, we cannot do cuda
    case DLDeviceType::kDLCUDA:
      return at::Device(DeviceType::CUDA, index);
#endif
    case DLDeviceType::kDLOpenCL:
      return at::Device(DeviceType::OPENCL, index);
    case DLDeviceType::kDLROCM:
#ifdef USE_ROCM
      // this looks funny, we need to return CUDA here to masquerade
      return at::Device(DeviceType::CUDA, index);
#else
      return at::Device(DeviceType::HIP, index);
#endif
    case DLDeviceType::kDLOneAPI:
      TORCH_CHECK(data != nullptr, "Can't get ATen device for XPU without XPU data.");
      return at::detail::getXPUHooks().getDeviceFromPtr(data);
    case DLDeviceType::kDLMAIA:
      return at::Device(DeviceType::MAIA, index);
    case DLDeviceType::kDLExtDev:
      return at::Device(DeviceType::PrivateUse1, index);
    case DLDeviceType::kDLMetal:
      return at::Device(DeviceType::MPS, index);
    default:
      TORCH_CHECK_BUFFER(
          false, "Unsupported device_type: ", std::to_string(type));
  }
}

ScalarType toScalarType(const DLDataType& dtype) {
  ScalarType stype = ScalarType::Undefined;
  if (dtype.code != DLDataTypeCode::kDLFloat4_e2m1fn) {
    TORCH_CHECK_BUFFER(
        dtype.lanes == 1,
        "ATen does not support lanes != 1 for dtype code", std::to_string(dtype.code));
  }
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Byte;
          break;
        case 16:
          stype = ScalarType::UInt16;
          break;
        case 32:
          stype = ScalarType::UInt32;
          break;
        case 64:
          stype = ScalarType::UInt64;
          break;
        default:
          TORCH_CHECK_BUFFER(
              false, "Unsupported kUInt bits ", std::to_string(dtype.bits));
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
          TORCH_CHECK_BUFFER(
              false, "Unsupported kInt bits ", std::to_string(dtype.bits));
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
          TORCH_CHECK_BUFFER(
              false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLBfloat:
      switch (dtype.bits) {
        case 16:
          stype = ScalarType::BFloat16;
          break;
        default:
          TORCH_CHECK_BUFFER(
              false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLComplex:
      switch (dtype.bits) {
        case 32:
          stype = ScalarType::ComplexHalf;
          break;
        case 64:
          stype = ScalarType::ComplexFloat;
          break;
        case 128:
          stype = ScalarType::ComplexDouble;
          break;
        default:
          TORCH_CHECK_BUFFER(
              false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLBool:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Bool;
          break;
        default:
          TORCH_CHECK_BUFFER(
              false, "Unsupported kDLBool bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e5m2:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Float8_e5m2;
          break;
        default:
          TORCH_CHECK_BUFFER(
              false, "Unsupported kDLFloat8_e5m2 bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e5m2fnuz:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Float8_e5m2fnuz;
          break;
        default:
          TORCH_CHECK_BUFFER(
              false, "Unsupported kDLFloat8_e5m2fnuz bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e4m3fn:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Float8_e4m3fn;
          break;
        default:
          TORCH_CHECK_BUFFER(
              false, "Unsupported kDLFloat8_e4m3fn bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e4m3fnuz:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Float8_e4m3fnuz;
          break;
        default:
          TORCH_CHECK_BUFFER(
              false, "Unsupported kDLFloat8_e4m3fnuz bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e8m0fnu:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Float8_e8m0fnu;
          break;
        default:
          TORCH_CHECK_BUFFER(
              false, "Unsupported kDLFloat8_e8m0fnu bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat4_e2m1fn:
      switch (dtype.bits) {
        case 4:
          switch (dtype.lanes) {
            case 2:
              stype = ScalarType::Float4_e2m1fn_x2;
              break;
            default:
              TORCH_CHECK_BUFFER(
                false, "Unsupported kDLFloat4_e2m1fn lanes ", std::to_string(dtype.lanes));
          }
          break;
        default:
          TORCH_CHECK_BUFFER(
              false, "Unsupported kDLFloat4_e2m1fn bits ", std::to_string(dtype.bits));
      }
      break;
    default:
      TORCH_CHECK_BUFFER(false, "Unsupported code ", std::to_string(dtype.code));
  }
  return stype;
}


namespace {

int64_t toStorageOffset(int64_t byte_offset, ScalarType stype) {
  if (byte_offset == 0) {
    return 0;
  }
  const auto element_size = c10::elementSize(stype);
  TORCH_CHECK_VALUE(byte_offset % element_size == 0, "byte offset must be multiple of element size");
  return byte_offset / element_size;
}

// The templated classes below are needed for supporting both:
//   - DLManagedTensor
//   - DLManagedTensorVersioned
template <class T>
struct ATenDLMTensor {
  Tensor handle;
  T tensor{};
};

template <class T>
void deleter(T* arg) {
  delete static_cast<ATenDLMTensor<T>*>(arg->manager_ctx);
}

// Adds version information for DLManagedTensorVersioned.
// This is a no-op for the other types.
template <class T>
void fillVersion(T* tensor) {}

template <>
void fillVersion<DLManagedTensorVersioned>(
    DLManagedTensorVersioned* tensor) {
  tensor->flags = 0;
  tensor->version.major = DLPACK_MAJOR_VERSION;
  tensor->version.minor = DLPACK_MINOR_VERSION;
}

// This function returns a shared_ptr to memory managed DLpack tensor
// constructed out of ATen tensor
template <class T>
T* toDLPackImpl(const Tensor& src) {
  auto atDLMTensor = std::make_unique<ATenDLMTensor<T>>();
  atDLMTensor->handle = src;
  atDLMTensor->tensor.manager_ctx = atDLMTensor.get();
  atDLMTensor->tensor.deleter = &deleter<T>;
  if (src.device().type()  == kMPS) {
      atDLMTensor->tensor.dl_tensor.data = src.storage().mutable_data();
      atDLMTensor->tensor.dl_tensor.byte_offset = src.storage_offset() * c10::elementSize(src.scalar_type());
  } else {
      atDLMTensor->tensor.dl_tensor.data = src.data_ptr();
      atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  }
  atDLMTensor->tensor.dl_tensor.device = torchDeviceToDLDevice(src.device());
  atDLMTensor->tensor.dl_tensor.ndim = static_cast<int32_t>(src.dim());
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataType(src);
  atDLMTensor->tensor.dl_tensor.shape = const_cast<int64_t*>(src.sizes().data());
  atDLMTensor->tensor.dl_tensor.strides = const_cast<int64_t*>(src.strides().data());
  fillVersion(&atDLMTensor->tensor);

  return &(atDLMTensor.release()->tensor);
}

// Explicitly instantiate the template above for both classes.
template DLManagedTensor* toDLPackImpl<DLManagedTensor>(const Tensor&);
template DLManagedTensorVersioned* toDLPackImpl<DLManagedTensorVersioned>(const Tensor&);

// This function constructs a Tensor from a memory managed DLPack which
// may be represented as either: DLManagedTensor and DLManagedTensorVersioned.
template <class T>
at::Tensor fromDLPackImpl(T* src, std::function<void(void*)> deleter) {
  if (!deleter) {
    deleter = [src](void* self [[maybe_unused]]) {
      if (src->deleter) {
        src->deleter(src);
      }
    };
  }

  DLTensor& dl_tensor = src->dl_tensor;
  Device device = dlDeviceToTorchDevice(
      dl_tensor.device.device_type, dl_tensor.device.device_id, dl_tensor.data);
  ScalarType stype = toScalarType(dl_tensor.dtype);

  if (!dl_tensor.strides) {
    TORCH_CHECK_VALUE(dl_tensor.byte_offset == 0, "Expected zero byte_offset");
    return at::from_blob(
        dl_tensor.data,
        IntArrayRef(dl_tensor.shape, dl_tensor.ndim),
        std::move(deleter),
        at::device(device).dtype(stype),
        {device});
  }
  return at::from_blob(
      dl_tensor.data,
      IntArrayRef(dl_tensor.shape, dl_tensor.ndim),
      IntArrayRef(dl_tensor.strides, dl_tensor.ndim),
      toStorageOffset(dl_tensor.byte_offset, stype),
      deleter,
      at::device(device).dtype(stype),
      {device});
}

// Explicitly instantiate the template above for both classes.
template at::Tensor fromDLPackImpl<DLManagedTensor>(DLManagedTensor* src, std::function<void(void*)> deleter);
template at::Tensor fromDLPackImpl<DLManagedTensorVersioned>(DLManagedTensorVersioned* src, std::function<void(void*)> deleter);

} // namespace

void toDLPackNonOwning(const Tensor& src, DLTensor* out) {
  // Fill in the pre-allocated DLTensor struct with direct pointers
  // This is a non-owning conversion - the caller owns the tensor
  // and must keep it alive for the duration of DLTensor usage
  out->data = src.data_ptr();
  out->device = torchDeviceToDLDevice(src.device());
  out->ndim = static_cast<int32_t>(src.dim());
  out->dtype = getDLDataType(src);
  // sizes() and strides() return pointers to TensorImpl's stable storage
  // which remains valid as long as the tensor is alive
  out->shape = const_cast<int64_t*>(src.sizes().data());
  out->strides = const_cast<int64_t*>(src.strides().data());
  out->byte_offset = 0;
}

DLManagedTensor* toDLPack(const Tensor& src) {
  return toDLPackImpl<DLManagedTensor>(src);
}

DLManagedTensorVersioned* toDLPackVersioned(const Tensor& src) {
  return toDLPackImpl<DLManagedTensorVersioned>(src);
}

Tensor fromDLPack(DLManagedTensor* src, std::function<void(void*)> deleter) {
  return fromDLPackImpl<DLManagedTensor>(src, std::move(deleter));
}

Tensor fromDLPackVersioned(DLManagedTensorVersioned* src, std::function<void(void*)> deleter) {
  return fromDLPackImpl<DLManagedTensorVersioned>(src, std::move(deleter));
}

Tensor maybeCopyTensor(
    const Tensor& data,
    std::optional<DLDevice> optional_dl_device,
    std::optional<bool> copy) {
  bool force_copy = copy.has_value() && *copy;
  bool force_move = copy.has_value() && !*copy;

  if (optional_dl_device.has_value()) {
    auto device = at::dlDeviceToTorchDevice(
        optional_dl_device->device_type,
        static_cast<c10::DeviceIndex>(optional_dl_device->device_id));

    if (device != data.device()) {
      TORCH_CHECK_VALUE(
          !force_move,
          "cannot move (i.e. copy=False) tensor from ",
          data.device(),
          " to ",
          device,
          " without copying.");
      return data.to(device);
    }
  }

  if (force_copy) {
    return data.clone();
  }

  return data;
}

} // namespace at

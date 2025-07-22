#include <ATen/DLConvertor.h>
#include <ATen/Functions.h>

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
    case ScalarType::Float8_e5m2fnuz:
    case ScalarType::Float8_e4m3fn:
    case ScalarType::Float8_e4m3fnuz:
    case ScalarType::Float8_e8m0fnu:
      TORCH_CHECK_BUFFER(false, "float8 types are not supported by dlpack");
      break;
    case ScalarType::Float4_e2m1fn_x2:
      TORCH_CHECK_BUFFER(false, "float4 types are not supported by dlpack");
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
    default:
      TORCH_CHECK_BUFFER(false, "Cannot pack tensors on " + device.str());
  }

  return ctx;
}

static Device getATenDevice(DLDeviceType type, c10::DeviceIndex index, void* data = nullptr) {
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
    default:
      TORCH_CHECK_BUFFER(
          false, "Unsupported device_type: ", std::to_string(type));
  }
}

ScalarType toScalarType(const DLDataType& dtype) {
  ScalarType stype = ScalarType::Undefined;
  TORCH_CHECK_BUFFER(dtype.lanes == 1, "ATen does not support lanes != 1");
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
    default:
      TORCH_CHECK_BUFFER(false, "Unsupported code ", std::to_string(dtype.code));
  }
  return stype;
}

namespace {

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
  // create a new tensor with possibly normalized strides
  // gh-83069
  auto shape = src.sizes();
  auto strides = src.strides().vec();
  for (int i = 0; i < src.dim(); i++) {
    if (shape[i] < 2) {
      strides[i] = 1;
    }
  }

  auto view = src.as_strided(shape, strides, src.storage_offset());
  ATenDLMTensor<T>* atDLMTensor(new ATenDLMTensor<T>);
  atDLMTensor->handle = view;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.deleter = &deleter<T>;
  atDLMTensor->tensor.dl_tensor.data = view.data_ptr();
  atDLMTensor->tensor.dl_tensor.device = torchDeviceToDLDevice(src.device());
  atDLMTensor->tensor.dl_tensor.ndim = static_cast<int32_t>(src.dim());
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataType(src);
  atDLMTensor->tensor.dl_tensor.shape = view.sizes().data();
  atDLMTensor->tensor.dl_tensor.strides = view.strides().data();
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  fillVersion(&atDLMTensor->tensor);

  return &(atDLMTensor->tensor);
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
  Device device = getATenDevice(dl_tensor.device.device_type, dl_tensor.device.device_id, dl_tensor.data);
  ScalarType stype = toScalarType(dl_tensor.dtype);

  if (!dl_tensor.strides) {
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
      deleter,
      at::device(device).dtype(stype),
      {device});
}

// Explicitly instantiate the template above for both classes.
template at::Tensor fromDLPackImpl<DLManagedTensor>(DLManagedTensor* src, std::function<void(void*)> deleter);
template at::Tensor fromDLPackImpl<DLManagedTensorVersioned>(DLManagedTensorVersioned* src, std::function<void(void*)> deleter);

} // namespace

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
    auto device = at::getATenDevice(
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

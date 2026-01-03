#include <torch/nativert/graph/TensorMeta.h>

namespace torch::nativert {

c10::ScalarType convertJsonScalarType(
    const torch::_export::ScalarType& scalarType) {
  switch (scalarType) {
    case torch::_export::ScalarType::UNKNOWN:
      TORCH_CHECK(false, "scalar type is not properly set");
    case torch::_export::ScalarType::BYTE:
      return c10::ScalarType::Byte;
    case torch::_export::ScalarType::CHAR:
      return c10::ScalarType::Char;
    case torch::_export::ScalarType::SHORT:
      return c10::ScalarType::Short;
    case torch::_export::ScalarType::INT:
      return c10::ScalarType::Int;
    case torch::_export::ScalarType::LONG:
      return c10::ScalarType::Long;
    case torch::_export::ScalarType::HALF:
      return c10::ScalarType::Half;
    case torch::_export::ScalarType::FLOAT:
      return c10::ScalarType::Float;
    case torch::_export::ScalarType::DOUBLE:
      return c10::ScalarType::Double;
    case torch::_export::ScalarType::COMPLEXHALF:
      return c10::ScalarType::ComplexHalf;
    case torch::_export::ScalarType::COMPLEXFLOAT:
      return c10::ScalarType::ComplexFloat;
    case torch::_export::ScalarType::COMPLEXDOUBLE:
      return c10::ScalarType::ComplexDouble;
    case torch::_export::ScalarType::BOOL:
      return c10::ScalarType::Bool;
    case torch::_export::ScalarType::BFLOAT16:
      return c10::ScalarType::BFloat16;
    case torch::_export::ScalarType::UINT16:
      return c10::ScalarType::UInt16;
    case torch::_export::ScalarType::FLOAT8E4M3FN:
      return c10::ScalarType::Float8_e4m3fn;
    case torch::_export::ScalarType::FLOAT8E5M2:
      return c10::ScalarType::Float8_e5m2;
    case torch::_export::ScalarType::FLOAT8E4M3FNUZ:
      return c10::ScalarType::Float8_e4m3fnuz;
    case torch::_export::ScalarType::FLOAT8E5M2FNUZ:
      return c10::ScalarType::Float8_e5m2fnuz;
    default:
      TORCH_CHECK(false, "unknown scalar type", static_cast<int>(scalarType));
  }
}

c10::MemoryFormat convertJsonMemoryFormat(
    const torch::_export::MemoryFormat& memoryFormat) {
  switch (memoryFormat) {
    case torch::_export::MemoryFormat::Unknown:
      TORCH_CHECK(false, "got unknown scalar type");
    case torch::_export::MemoryFormat::ContiguousFormat:
      return c10::MemoryFormat::Contiguous;
    case torch::_export::MemoryFormat::ChannelsLast:
      return c10::MemoryFormat::ChannelsLast;
    case torch::_export::MemoryFormat::ChannelsLast3d:
      return c10::MemoryFormat::ChannelsLast3d;
    case torch::_export::MemoryFormat::PreserveFormat:
      return c10::MemoryFormat::Preserve;
    default:
      TORCH_CHECK(
          false, "unknown memory format", static_cast<int>(memoryFormat));
  }
}

c10::Layout convertJsonLayout(const torch::_export::Layout& layout) {
  switch (layout) {
    case torch::_export::Layout::Unknown:
      TORCH_CHECK(false, "got unknown layout");
    case torch::_export::Layout::SparseCoo:
      // TODO is this the right translation
      return c10::Layout::Sparse;
    case torch::_export::Layout::SparseCsr:
      return c10::Layout::SparseCsr;
    case torch::_export::Layout::SparseCsc:
      return c10::Layout::SparseCsc;
    case torch::_export::Layout::SparseBsr:
      return c10::Layout::SparseBsr;
    case torch::_export::Layout::SparseBsc:
      return c10::Layout::SparseBsc;
    case torch::_export::Layout::_mkldnn:
      return c10::Layout::Mkldnn;
    case torch::_export::Layout::Strided:
      return c10::Layout::Strided;
    default:
      TORCH_CHECK(false, "unknown layout", static_cast<int>(layout));
  }
}

c10::Device convertJsonDevice(const torch::_export::Device& device) {
  c10::Device d(device.get_type());
  if (auto index = device.get_index()) {
    d.set_index(static_cast<at::DeviceIndex>(*index));
  }
  return d;
}

TensorMeta::TensorMeta(const torch::_export::TensorMeta& tensorMeta)
    : dtype_(convertJsonScalarType(tensorMeta.get_dtype())),
      layout_(convertJsonLayout(tensorMeta.get_layout())),
      requiresGrad_(tensorMeta.get_requires_grad()),
      device_(convertJsonDevice(tensorMeta.get_device())) {
  const auto& storageOffset = tensorMeta.get_storage_offset();
  if (storageOffset.tag() == torch::_export::SymInt::Tag::AS_INT) {
    storage_offset_ = tensorMeta.get_storage_offset().get_as_int();
  } else if (storageOffset.tag() == torch::_export::SymInt::Tag::AS_EXPR) {
    // TODO: it's still unclear how SymInt shape should be used in runtime
    // setting the storage offset to 0 for now
    hasSymbolicShape_ = true;
    storage_offset_ = 0;
  }

  for (const auto& size : tensorMeta.get_sizes()) {
    if (size.tag() == torch::_export::SymInt::Tag::AS_INT) {
      int64_t val = size.get_as_int();
      sizes_.emplace_back(val);
      numel_ *= val;
    } else if (size.tag() == torch::_export::SymInt::Tag::AS_EXPR) {
      // TODO: it's still unclear how SymInt shape should be used in runtime
      // One potential use cases is for verifying inputs shape matches constrain
      // This would require unpacking the serialized constrain, which is NYI
      //
      // For the time being, we just set the symbolic dim to -1
      hasSymbolicShape_ = true;
      sizes_.emplace_back(-1);
      numel_ = -1;
    }
  }

  for (const auto& stride : tensorMeta.get_strides()) {
    if (stride.tag() == torch::_export::SymInt::Tag::AS_INT) {
      strides_.emplace_back(stride.get_as_int());
    } else if (stride.tag() == torch::_export::SymInt::Tag::AS_EXPR) {
      // TODO: it's still unclear how SymInt shape should be used in runtime
      // Setting symbolic shape to -1 for now
      hasSymbolicShape_ = true;
      strides_.emplace_back(-1);
    }
  }
}

} // namespace torch::nativert

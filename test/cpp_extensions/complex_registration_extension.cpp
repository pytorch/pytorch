#include <torch/extension.h>

#include <ATen/CPUFloatType.h>
#include <ATen/Type.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/detail/ComplexHooksInterface.h>

#include <c10/core/Allocator.h>
#include <ATen/CPUGenerator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/Half.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>

namespace at {

struct CPUComplexFloatType : public at::CPUTypeDefault {
  CPUComplexFloatType()
      : CPUTypeDefault(
            CPUTensorId(),
            /*is_variable=*/false,
            /*is_undefined=*/false) {}

  ScalarType scalarType() const override;
  caffe2::TypeMeta typeMeta() const override;
  Backend backend() const override;
  const char* toString() const override;
  size_t elementSizeInBytes() const override;
  TypeID ID() const override;

  Tensor empty(IntArrayRef size, const TensorOptions & options) const override {
    // Delegate to the appropriate cpu tensor factory
    const DeviceGuard device_guard(options.device());
    return at::native::empty_cpu(/* actuals */ size, options);
  }
};

struct ComplexHooks : public at::ComplexHooksInterface {
  ComplexHooks(ComplexHooksArgs) {}
  void registerComplexTypes(Context* context) const override {
    context->registerType(
        Backend::CPU, ScalarType::ComplexFloat, new CPUComplexFloatType());
  }
};

ScalarType CPUComplexFloatType::scalarType() const {
  return ScalarType::ComplexFloat;
}

caffe2::TypeMeta CPUComplexFloatType::typeMeta() const {
  return scalarTypeToTypeMeta(ScalarType::ComplexFloat);
}

Backend CPUComplexFloatType::backend() const {
  return Backend::CPU;
}

const char* CPUComplexFloatType::toString() const {
  return "CPUComplexFloatType";
}

TypeID CPUComplexFloatType::ID() const {
  return TypeID::CPUComplexFloat;
}

size_t CPUComplexFloatType::elementSizeInBytes() const {
  return sizeof(float);
}

REGISTER_COMPLEX_HOOKS(ComplexHooks);

} // namespace at

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { }

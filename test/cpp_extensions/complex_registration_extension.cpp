#include <torch/extension.h>

#include <ATen/Type.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/detail/ComplexHooksInterface.h>

#include <ATen/CPUTypeDefault.h>
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

struct ComplexCPUType : public at::CPUTypeDefault {
  ComplexCPUType()
      : CPUTypeDefault(
            ComplexCPUTensorId(),
            /*is_variable=*/false,
            /*is_undefined=*/false) {}

  Backend backend() const override;
  const char* toString() const override;
  TypeID ID() const override;

  Tensor empty(IntArrayRef size, const TensorOptions & options) const override {
    AT_ASSERT(options.device().is_cpu());

    for (auto x: size) {
      TORCH_CHECK(x >= 0, "Trying to create tensor using size with negative dimension: ", size);
    }
    auto* allocator = at::getCPUAllocator();
    int64_t nelements = at::prod_intlist(size);
    auto dtype = options.dtype();
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        dtype,
        nelements,
        allocator->allocate(nelements * dtype.itemsize()),
        allocator,
        /*resizable=*/true);

    auto tensor = detail::make_tensor<TensorImpl>(storage_impl, at::ComplexCPUTensorId());
    // Default TensorImpl has size [0]
    if (size.size() != 1 || size[0] != 0) {
      tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }
    return tensor;
  }
};

struct ComplexHooks : public at::ComplexHooksInterface {
  ComplexHooks(ComplexHooksArgs) {}
  void registerComplexTypes(Context* context) const override {
    context->registerType(Backend::ComplexCPU, new ComplexCPUType());
  }
};

Backend ComplexCPUType::backend() const {
  return Backend::ComplexCPU;
}

const char* ComplexCPUType::toString() const {
  return "ComplexCPUType";
}

TypeID ComplexCPUType::ID() const {
  return TypeID::ComplexCPU;
}

REGISTER_COMPLEX_HOOKS(ComplexHooks);

} // namespace at

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { }

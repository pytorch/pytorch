#include <c10/util/Exception.h>
#include <operator_registry.h>

namespace torch {
namespace executor {

KernelRegistry& getKernelRegistry() {
  static KernelRegistry kernel_registry;
  return kernel_registry;
}

bool register_kernels(const ArrayRef<Kernel>& kernels) {
  return getKernelRegistry().register_kernels(kernels);
}

bool KernelRegistry::register_kernels(
    const ArrayRef<Kernel>& kernels) {
  for (const auto& kernel : kernels) {
    this->kernels_map_[kernel.name_] = kernel.kernel_;
  }
  return true;
}

bool hasKernelFn(const char* name) {
  return getKernelRegistry().hasKernelFn(name);
}

bool KernelRegistry::hasKernelFn(const char* name) {
  auto kernel = this->kernels_map_.find(name);
  return kernel != this->kernels_map_.end();
}

KernelFunction& getKernelFn(const char* name) {
  return getKernelRegistry().getKernelFn(name);
}

KernelFunction& KernelRegistry::getKernelFn(const char* name) {
  auto kernel = this->kernels_map_.find(name);
  TORCH_CHECK_MSG(kernel != this->kernels_map_.end(), "Kernel not found!");
  return kernel->second;
}


} // namespace executor
} // namespace torch

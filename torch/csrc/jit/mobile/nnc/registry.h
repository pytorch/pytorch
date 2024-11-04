#pragma once

#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

namespace torch::jit::mobile::nnc {

using nnc_kernel_function_type = int(void**);

struct TORCH_API NNCKernel {
  virtual ~NNCKernel() = default;
  virtual int execute(void** /* args */) = 0;
};

TORCH_DECLARE_REGISTRY(NNCKernelRegistry, NNCKernel);

#define REGISTER_NNC_KERNEL(id, kernel, ...)     \
  extern "C" {                                   \
  nnc_kernel_function_type kernel;               \
  }                                              \
  struct NNCKernel_##kernel : public NNCKernel { \
    int execute(void** args) override {          \
      return kernel(args);                       \
    }                                            \
  };                                             \
  C10_REGISTER_TYPED_CLASS(NNCKernelRegistry, id, NNCKernel_##kernel);

namespace registry {

inline bool has_nnc_kernel(const std::string& id) {
  return NNCKernelRegistry()->Has(id);
}

inline std::unique_ptr<NNCKernel> get_nnc_kernel(const std::string& id) {
  return NNCKernelRegistry()->Create(id);
}

} // namespace registry

} // namespace torch::jit::mobile::nnc

#include <ATen/core/dispatch/Operator.h>

namespace c10 {
namespace impl {

Operator::Operator(FunctionSchema&& schema)
: schema_(std::move(schema))
, kernels_(schema_) {}

void Operator::prepareForDeregistration() {
  return kernels_.read([&] (const Kernels& kernels) {
    if (!kernels.dispatchTable.isEmpty()) {
      std::ostringstream str;
      str << schema_;
      AT_ERROR("Tried to deregister op schema for an operator that still has kernels registered. The operator schema is ", str.str());
    }
  });
}

RegistrationHandleRAII Operator::registerKernel(TensorTypeId dispatch_key, DispatchTableEntry kernel) {
  kernels_.write([&] (Kernels& kernels) {
    kernels.dispatchTable.registerKernel(dispatch_key, std::move(kernel));
  });

  return RegistrationHandleRAII([this, dispatch_key] {
    deregisterKernel_(dispatch_key);
  });
}

void Operator::deregisterKernel_(TensorTypeId dispatch_key) {
  kernels_.write([&] (Kernels& kernels) {
    kernels.dispatchTable.deregisterKernel(std::move(dispatch_key));
  });
}

RegistrationHandleRAII Operator::registerFallbackKernel(DispatchTableEntry kernel) {
  kernels_.write([&] (Kernels& kernels) {
    kernels.dispatchTable.registerFallbackKernel(std::move(kernel));
  });

  return RegistrationHandleRAII([this] {
    deregisterFallbackKernel_();
  });
}

void Operator::deregisterFallbackKernel_() {
  kernels_.write([&] (Kernels& kernels) {
    kernels.dispatchTable.deregisterFallbackKernel();
  });
}

}
}

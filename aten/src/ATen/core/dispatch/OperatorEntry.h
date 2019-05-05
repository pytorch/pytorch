#pragma once

#include <ATen/core/dispatch/DispatchTable.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>

namespace c10 {
namespace impl {

// This is a private class used inside the Dispatcher to represent an operator
// and it's dispatch table. This is not part of the public API.
class OperatorEntry final {
private:
  struct Kernels final {
    DispatchTable dispatchTable;
  };
public:
  explicit OperatorEntry(FunctionSchema&& schema);

  OperatorEntry(const OperatorEntry&) = delete;
  OperatorEntry(OperatorEntry&&) noexcept = delete;
  OperatorEntry& operator=(const OperatorEntry&) = delete;
  OperatorEntry& operator=(OperatorEntry&&) noexcept = delete;

  const FunctionSchema& schema() const {
    return schema_;
  }

  DispatchTableEntry lookupKernel(const Stack* stack) const {
    return kernels_.read([&] (const Kernels& kernels) {
      return kernels.dispatchTable.lookup(stack);
    });
  }

  void prepareForDeregistration();

  RegistrationHandleRAII registerKernel(TensorTypeId dispatch_key, DispatchTableEntry kernel);
  RegistrationHandleRAII registerFallbackKernel(DispatchTableEntry kernel);

private:
  void deregisterKernel_(TensorTypeId dispatch_key);
  void deregisterFallbackKernel_();

  FunctionSchema schema_;
  LeftRight<Kernels> kernels_;
};

}
}

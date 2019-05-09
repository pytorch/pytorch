#pragma once

/**
 * This file implements c10::dispatchKey() which is used in the kernel
 * registration API to set the dispatch key for a registered kernel.
 *
 * You probably don't want to include this file directly but include
 * op_registration.h instead since that adds more functionality you'll
 * likely need to register your operators.
 */

#include <ATen/core/op_registration/base.h>

namespace c10 {

namespace detail {
  struct DispatchKeyConfigParameter final {
    explicit constexpr DispatchKeyConfigParameter(TensorTypeId dispatch_key)
    : dispatch_key_(dispatch_key) {}

    void apply(KernelRegistrationConfig* registration) const {
      registration->dispatch_key = dispatch_key_;
    }

  private:
    TensorTypeId dispatch_key_;
  };
  static_assert(is_registration_config_parameter<DispatchKeyConfigParameter>::value, "KernelRegistrationConfigParameter must fulfill the registration config parameter concept");
}

/**
 * Use this to register an operator with a kernel for a certain dispatch key.
 *
 * Example:
 *
 * > namespace {
 * >   class my_kernel_cpu final : public c10::OperatorKernel {
 * >   public:
 * >     Tensor operator()(Tensor a, Tensor b) {...}
 * >   };
 * >   class my_kernel_cuda final : public c10::OperatorKernel {
 * >   public:
 * >     Tensor operator()(Tensor a, Tensor b) {...}
 * >   };
 * > }
 * >
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op",
 * >         c10::kernel<my_kernel_cpu>(),
 * >         c10::dispatchKey(CPUTensorId()))
 * >     .op("my_op",
 * >         c10::kernel<my_kernel_cuda>(),
 * >         c10::dispatchKey(CUDATensorId()));
 */
inline constexpr detail::DispatchKeyConfigParameter dispatchKey(TensorTypeId dispatch_key) {
  return detail::DispatchKeyConfigParameter(dispatch_key);
}

}

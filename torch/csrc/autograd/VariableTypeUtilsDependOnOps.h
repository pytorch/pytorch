#pragma once

#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/jit_decomp_interface.h>

// This is the set of helpers in VariableTypeUtils have a dependency on
// native_functions.yaml meaning the file will need to be re-compiled every time
// an operator is changed or added. We cannot simply put these functions in
// VariableType.h and VariableTypeutils.h, since they are included in files like
// ADInplaceOrViewType_X.cpp which don't always want to be recompiled.

namespace torch {
namespace autograd {
namespace impl {

class MyFunctor final : public c10::OperatorKernel {
 public:
  MyFunctor(JitDecompInterface* fns) : fns_(fns){};

  void operator()(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet ks,
      torch::jit::Stack* stack) {
    fns_->run_jit_decomposition_(op, stack);
  }
  JitDecompInterface* fns_;
};

// Depends on torch/csrc/jit/ir/ir.h -> aten/src/ATen/core/interned_strings.h
template <class Return, class... Args>
Return run_jit_decomposition_with_args_for_jvp(
    c10::string_view name,
    const c10::OperatorHandle& opHandle,
    c10::DispatchKeySet dispatchKeySet,
    Args&&... args) {
  JitDecompInterface* fns = getJitDecomp();
  bool has_decomp = fns->has_jit_decomposition_(opHandle.schema());

  TORCH_CHECK_NOT_IMPLEMENTED(
      has_decomp,
      "Trying to use forward AD with ",
      name,
      " that does not support it "
      "because it has not been implemented yet and does not have a decomposition.\nPlease file an issue "
      "to PyTorch at https://github.com/pytorch/pytorch/issues/new?template=feature-request.yml "
      "so that we can prioritize its implementation.\n"
      "Note that forward AD support for some operators require JIT to be enabled. If the environment var "
      "PYTORCH_JIT=0 is set, some operators may no longer be used with forward AD.");

  return c10::KernelFunction::makeFromBoxedKernel(
             c10::BoxedKernel::makeFromFunctor(
                 std::make_unique<MyFunctor>(fns)))
      .call<Return, Args...>(
          opHandle, dispatchKeySet, std::forward<Args>(args)...);
}

} // namespace impl
} // namespace autograd
} // namespace torch

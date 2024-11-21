#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/function_schema.h>
#include <c10/macros/Export.h>

// NOTE: [Jit Decomposition Interface]
//
// For some context of why we need this at all, see NOTE: [forward-mode AD
// decompositions mechanism]
//
// Introducing that mechanism from the NOTE is problematic because:
// - it relies on TorchScript, so now VariableTypeX.cpp depends on TorchScript.
// - there exist internal builds like lite_trainer, which depend on VariableType
//   but do not depend on TorchScript.
//
// For internal builds like lite_trainer builds to pass, and for OSS builds that
// do depend on TorchScript to still support the forward AD decomp mechanism, we
// implement a PImpl pattern to avoid a static dependency in favor of a dynamic
// one
// - during static initialization time, if the library is built with TorchScript
//   setJitDecompImpl is called in decomposition_registry.cpp setting a global
//   ptr to the impl
// - when the program is run,if getJitDecompImpl returns a non null ptr, we can
//   carry on normally, otherwise we gracefully error out
//
// For extra context, see VariableHooksInterface.h, where a similar technique
// is used

namespace torch::autograd::impl {

struct TORCH_API JitDecompInterface {
  virtual ~JitDecompInterface() = default;
  virtual bool has_jit_decomposition(
      const c10::FunctionSchema& schema) const = 0;
  virtual void run_jit_decomposition(
      const c10::OperatorHandle& op,
      jit::Stack* stack) const = 0;
};

TORCH_API void setJitDecompImpl(JitDecompInterface* impl);
TORCH_API JitDecompInterface* getJitDecompImpl();

struct TORCH_API JitDecompRegisterer {
  explicit JitDecompRegisterer(JitDecompInterface* impl) {
    setJitDecompImpl(impl);
  }
};

} // namespace torch::autograd::impl

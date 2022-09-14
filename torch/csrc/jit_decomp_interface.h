#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/function_schema.h>
#include <c10/macros/Export.h>

namespace torch {
namespace autograd {
namespace impl {

struct TORCH_API JitDecompInterface {
  virtual ~JitDecompInterface() = default;
  virtual bool has_jit_decomposition_(
      const c10::FunctionSchema& schema) const = 0;
  virtual void run_jit_decomposition_(
      const c10::OperatorHandle& op,
      jit::Stack* stack) const = 0;
};

TORCH_API void setJitDecompInterface(JitDecompInterface* fns);
TORCH_API JitDecompInterface* getJitDecomp();

struct TORCH_API JitDecompRegisterer {
  explicit JitDecompRegisterer(JitDecompInterface* fns) {
    setJitDecompInterface(fns);
  }
};

} // namespace impl
} // namespace autograd
} // namespace torch

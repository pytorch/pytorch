#include "torch/csrc/nativert/kernels/KernelRegistry.h"

namespace torch::nativert {

REGISTER_CPU_KERNEL("torch.ops.aten.slice.Tensor", aten_slice_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& dim = KernelInput(1).toInt();
  const auto& start = KernelInput(2).toOptional<int64_t>();
  const auto& end = KernelInput(3).toOptional<int64_t>();
  const auto& step = KernelInput(4).toInt();
  KernelOutput(0) = at::native::slice(self, dim, start, end, step);
});

} // namespace torch::nativert

#include <torch/nativert/kernels/KernelRegistry.h>

#include <ATen/native/NonSymbolicBC.h>

namespace torch::nativert {

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.slice.Tensor", aten_slice_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& dim = KernelInput(1).toInt();
  const auto& start = KernelInput(2).toOptional<int64_t>();
  const auto& end = KernelInput(3).toOptional<int64_t>();
  const auto& step = KernelInput(4).toInt();
  KernelOutput(0) = at::native::slice(self, dim, start, end, step);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.sym_size.int", aten_sym_size_int, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  auto& out = KernelOutput(0);
  TORCH_CHECK(dim >= 0 && dim < self.dim(), "Invalid dimension");
  out = self.sym_size(dim);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.reshape.default", aten_reshape, {
  const auto& self = KernelInput(0).toTensor();
  const auto& shape = KernelInput(1).toIntVector();
  KernelOutput(0) = at::native::reshape(self, shape);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.view.default", aten_view, {
  const auto& self = KernelInput(0).toTensor();
  const auto& size = KernelInput(1).toIntVector();
  KernelOutput(0) = at::native::view(self, size);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.permute.default", aten_permute, {
  const auto& self = KernelInput(0).toTensor();
  const auto& dims = KernelInput(1).toDimVector();
  KernelOutput(0) = at::native::permute(self, dims);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.select.int", aten_select, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto index = KernelInput(2).toInt();
  KernelOutput(0) = at::native::select(self, dim, index);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.split.Tensor", aten_split_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto split_size = KernelInput(1).toInt();
  const auto dim = KernelInput(2).toInt();
  KernelOutput(0) = at::native::split(self, split_size, dim);
})

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.split_with_sizes.default",
    aten_split_with_sizes,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& split_sizes = KernelInput(1).toIntList();
      const auto dim = KernelInput(2).toInt();
      KernelOutput(0) =
          at::native::split_with_sizes(self, split_sizes.vec(), dim);
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.tensor_split.sections",
    aten_tensor_split_sections,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto sections = KernelInput(1).toInt();
      const auto dim = KernelInput(2).toInt();
      KernelOutput(0) =
          at::native::tensor_split_sections_symint(self, sections, dim);
    })

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.item.default", aten_item, {
  const auto& self = KernelInput(0).toTensor();
  KernelOutput(0) = at::native::item(self);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.narrow.default", aten_narrow, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  int64_t start = 0;
  if (KernelInput(2).isScalar()) {
    start = KernelInput(2).toInt();
  } else {
    auto& t = KernelInput(2).toTensor();
    start = t.item<int64_t>();
  }
  const auto length = KernelInput(3).toInt();
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.sizes()[dim];
  if (start != cur_size && start < 0) {
    start = at::maybe_wrap_dim(start, cur_size);
  }
  TORCH_CHECK(
      length >= 0 && start <= cur_size - length,
      "start (",
      start,
      ") + length (",
      length,
      ") exceeds dimension size (",
      cur_size,
      ").");
  KernelOutput(0) = at::native::slice(self, dim, start, start + length, 1);
})

} // namespace torch::nativert

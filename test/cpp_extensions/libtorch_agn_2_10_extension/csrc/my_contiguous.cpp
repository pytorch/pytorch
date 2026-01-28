#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

// Test contiguous with default memory format
Tensor my_contiguous(Tensor self) {
  return torch::stable::contiguous(self);
}

// Test contiguous with specified memory format
Tensor my_contiguous_memory_format(
    Tensor self,
    torch::headeronly::MemoryFormat memory_format) {
  return torch::stable::contiguous(self, memory_format);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my_contiguous(Tensor self) -> Tensor");
  m.def("my_contiguous_memory_format(Tensor self, MemoryFormat memory_format) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_contiguous", TORCH_BOX(&my_contiguous));
  m.impl("my_contiguous_memory_format", TORCH_BOX(&my_contiguous_memory_format));
}

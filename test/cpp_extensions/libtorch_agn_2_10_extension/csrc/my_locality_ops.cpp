#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

using torch::stable::Tensor;

namespace {

Tensor my_index_fill(Tensor self, int64_t dim, const Tensor& index, const Tensor& value) {
  return torch::stable::index_fill_(self, dim, index, value);
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_roll(Tensor self, int[] shifts, int[] dims) -> Tensor");
  m.def(
      "my_repeat_interleave(Tensor self, int repeats, int? dim=None, int? output_size=None) -> Tensor");
  m.def("my_equal(Tensor self, Tensor other) -> bool");
  m.def("my_nonzero(Tensor self) -> Tensor");
  m.def("my_expand(Tensor(a) self, int[] size, bool implicit=False) -> Tensor(a)");
  m.def("my_all(Tensor self) -> Tensor");
  m.def("my_median(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)");
  m.def("my_index_fill(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_roll", TORCH_BOX(&torch::stable::roll));
  m.impl("my_repeat_interleave", TORCH_BOX(&torch::stable::repeat_interleave));
  m.impl("my_equal", TORCH_BOX(&torch::stable::equal));
  m.impl("my_nonzero", TORCH_BOX(&torch::stable::nonzero));
  m.impl("my_expand", TORCH_BOX(&torch::stable::expand));
  m.impl("my_all", TORCH_BOX(&torch::stable::all));
  m.impl("my_median", TORCH_BOX(&torch::stable::median));
  m.impl("my_index_fill", TORCH_BOX(&my_index_fill));
}

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

uint64_t get_any_data_ptr(Tensor t, bool mutable_) {
  if (mutable_) {
    return reinterpret_cast<uint64_t>(t.mutable_data_ptr());
  } else {
    return reinterpret_cast<uint64_t>(t.const_data_ptr());
  }
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("get_any_data_ptr(Tensor t, bool mutable_) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("get_any_data_ptr", TORCH_BOX(&get_any_data_ptr));
}

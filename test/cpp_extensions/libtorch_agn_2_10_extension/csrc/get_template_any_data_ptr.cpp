#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

using torch::stable::Tensor;

uint64_t get_template_any_data_ptr(Tensor t, torch::headeronly::ScalarType dtype, bool mutable_) {
#define DEFINE_CASE(T, name)                                            \
  case torch::headeronly::ScalarType::name: {                           \
    if (mutable_) {                                                     \
      return reinterpret_cast<uint64_t>(t.mutable_data_ptr<T>());       \
    } else {                                                            \
      return reinterpret_cast<uint64_t>(t.const_data_ptr<T>());         \
    }                                                                   \
  }
  switch (dtype) {
    // per aten/src/ATen/templates/TensorMethods.cpp:
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
    DEFINE_CASE(uint16_t, UInt16)
    DEFINE_CASE(uint32_t, UInt32)
    DEFINE_CASE(uint64_t, UInt64)
  default:
      return 0;
  }
#undef DEFINE_CASE
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("get_template_any_data_ptr(Tensor t, ScalarType dtype, bool mutable_) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("get_template_any_data_ptr", TORCH_BOX(&get_template_any_data_ptr));
}

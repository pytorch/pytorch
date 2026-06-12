#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/Dispatch_v2.h>

using torch::stable::Tensor;

uint64_t get_template_any_data_ptr(Tensor t, torch::headeronly::ScalarType dtype, bool mutable_) {
#define DEFINE_CASE(T, name)                                            \
  case name: {                           \
    if (mutable_) {                                                     \
      return reinterpret_cast<uint64_t>(t.mutable_data_ptr<T>());       \
    } else {                                                            \
      return reinterpret_cast<uint64_t>(t.const_data_ptr<T>());         \
    }                                                                   \
  }
  switch (dtype) {
    // per aten/src/ATen/templates/TensorMethods.cpp:
    AT_FORALL_SCALAR_TYPES_V2(
      AT_WRAP(DEFINE_CASE),
      AT_EXPAND(AT_ALL_SCALAR_TYPES_WITH_COMPLEX),
      torch::headeronly::ScalarType::UInt16,
      torch::headeronly::ScalarType::UInt32,
      torch::headeronly::ScalarType::UInt64
    )
  default:
      return 0;
  }
#undef DEFINE_CASE
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("get_template_any_data_ptr(Tensor t, ScalarType dtype, bool mutable_) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("get_template_any_data_ptr", TORCH_BOX(&get_template_any_data_ptr));
}

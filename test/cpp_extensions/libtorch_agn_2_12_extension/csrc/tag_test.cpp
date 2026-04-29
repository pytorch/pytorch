#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/enum_tag.h>

using torch::stable::Tensor;

Tensor tagged_identity(Tensor t) {
  return t;
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("tagged_identity(Tensor t) -> Tensor",
        std::vector<at::Tag>{
            at::Tag::pointwise, at::Tag::pt2_compliant_tag, at::Tag::core});
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("tagged_identity", TORCH_BOX(&tagged_identity));
}

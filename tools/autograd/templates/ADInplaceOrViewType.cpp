#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <torch/library.h>


#include <ATen/RedispatchFunctions.h>

// ${generated_comment}


using namespace at;
using torch::autograd::CreationMeta;
using torch::autograd::as_view;
using torch::autograd::increment_version;

namespace torch {

namespace ADInplaceOrView {

namespace {
${inplace_or_view_method_definitions}
}  // namespace
}  // namespace ADInplaceOrView

namespace {

TORCH_LIBRARY_IMPL(aten, ADInplaceOrView, m) {
  ${inplace_or_view_wrapper_registrations};
}

}  // namespace
} // namespace torch

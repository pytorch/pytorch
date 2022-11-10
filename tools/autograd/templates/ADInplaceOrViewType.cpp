#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <torch/library.h>

// ${generated_comment}

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
$ops_headers
#endif

using namespace at;
using torch::autograd::CreationMeta;
using torch::autograd::as_view;
using torch::autograd::increment_version;
using torch::autograd::impl::maybe_copy_on_write;

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

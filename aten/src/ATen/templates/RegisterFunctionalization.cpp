#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/FunctionalInverses.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_strided_native.h>
$ops_headers
#endif

namespace at {
namespace functionalization {


${func_definitions}

namespace addbackviews {

${func_add_back_views_definitions}

} // namespace addbackviews

}  // namespace functionalization

namespace {

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  ${func_registrations};
}

TORCH_LIBRARY_IMPL(aten, FunctionalizeAddBackViews, m) {
  ${func_add_back_views_registrations};
}

}  // namespace

} // namespace at

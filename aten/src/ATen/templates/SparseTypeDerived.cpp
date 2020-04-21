// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include <ATen/${Type}.h>

// ${generated_comment}

#include <ATen/${Generator}.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/Dispatch.h>
#include <c10/util/Half.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/core/op_registration/op_registration.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
$extra_cuda_headers

namespace {
static const char* named_tensors_unsupported_error =
  " is not yet supported with named tensors. Please drop names via "
  "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
  "and set names on the result of the operation.";
}

namespace at {

namespace ${Type} {

${type_derived_method_definitions}

}  // namespace ${Type}

TORCH_LIBRARY_IMPL(aten, ${Backend}, m) {
  ${function_registrations};
}

} // namespace at

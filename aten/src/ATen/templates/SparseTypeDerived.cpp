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
#include <ATen/core/EnableNamedTensor.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
$extra_cuda_headers

namespace at {

${type_derived_method_definitions}

#ifndef USE_STATIC_DISPATCH
static auto& registerer = globalATenDispatch()
  ${function_registrations};
#endif
}

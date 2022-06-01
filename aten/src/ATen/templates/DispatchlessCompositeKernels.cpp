// ${generated_comment}
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/Dispatch.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Half.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/Tensor.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
${aggregated_headers}
#else
${operator_headers}
#endif

${kernel_headers}

namespace at {
namespace native {

${kernel_definitions}

} // namespace native
} // namespace at

#pragma once

// ${generated_comment}

#ifdef TORCH_ASSERT_ONLY_METHOD_OPERATORS
#error This change adds a dependency on all pytorch operators, meaning the     \
  file will need to be re-compiled every time an operator is changed or added. \
  Consider using the at::_ops::{name}::redispatch() interface by including     \
  the specific operator from <ATen/ops/{my_operator}_ops.h>
#endif

#include <c10/core/Scalar.h>
#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <ATen/core/Generator.h>
#include <c10/util/Deprecated.h>
#include <ATen/DeviceGuard.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/Reduction.h>
#include <c10/util/Optional.h>
#include <ATen/TensorUtils.h>
#include <ATen/Context.h>
#include <ATen/TracerMode.h>
#include <ATen/Operators.h>

namespace at {

namespace redispatch {
    ${function_redispatch_definitions}
} // namespace redispatch

}

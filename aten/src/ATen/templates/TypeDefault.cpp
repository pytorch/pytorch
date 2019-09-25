#include <ATen/TypeDefault.h>

// ${generated_comment}

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <ATen/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <ATen/DeviceGuard.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/core/ATenDispatch.h>
#include <ATen/core/EnableNamedTensor.h>

namespace at {

${type_method_definitions}

#ifndef USE_STATIC_DISPATCH
static auto& registerer = globalATenDispatch()
  ${function_registrations};
#endif
}

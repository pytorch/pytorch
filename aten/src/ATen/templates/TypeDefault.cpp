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
#include <ATen/core/op_registration/op_registration.h>

namespace at {

${type_method_definitions}

namespace {

static auto& registerer = globalATenDispatch()
  ${function_registrations};

static auto c10_registerer = torch::RegisterOperators()
  ${c10_function_registrations};

}

}

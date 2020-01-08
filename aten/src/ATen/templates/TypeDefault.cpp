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
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Device.h>
#include <ATen/DeviceGuard.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/core/op_registration/op_registration.h>

namespace at {
namespace TypeDefault {

${type_method_definitions}

}  // namespace TypeDefault

#ifndef USE_STATIC_DISPATCH
namespace {
auto registerer = torch::RegisterOperators()
  ${function_registrations};
}
#endif

}  // namespace at

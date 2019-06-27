#include <ATen/TypeDefault.h>

// ${generated_comment}

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#ifdef NAMEDTENSOR_ENABLED
#include <ATen/NamedTensorUtils.h>
#endif
#include <ATen/NativeFunctions.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <ATen/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <ATen/DeviceGuard.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/core/ATenDispatch.h>

namespace at {

void TypeDefault::backward(
    Tensor& self,
    c10::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) const {
  AT_ERROR("backward is not implemented for Tensor");
}

void TypeDefault::set_data(Tensor & self, Tensor new_data) const {
  AT_ERROR("set_data is not implemented for Tensor");
}

Type & TypeDefault::toBackend(Backend b) const {
  return at::globalContext().getNonVariableType(b, ScalarType::Undefined);
}
Type & TypeDefault::toScalarType(ScalarType s) const {
  return at::globalContext().getNonVariableType(backend(),s);
}

${type_method_definitions}

static auto& registerer = globalATenDispatch()
  ${function_registrations};
}

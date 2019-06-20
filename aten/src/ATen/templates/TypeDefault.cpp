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
#include <ATen/core/TensorOptions.h>
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

Tensor TypeDefault::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  auto tensor_impl = c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(static_cast<TensorImpl*>(th_pointer));
  if (retain && tensor_impl.get() != UndefinedTensorImpl::singleton()) {
    c10::raw::intrusive_ptr::incref(tensor_impl.get());
  }
  return Tensor(std::move(tensor_impl));
}
Storage TypeDefault::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain && th_pointer) {
    c10::raw::intrusive_ptr::incref(static_cast<StorageImpl*>(th_pointer));
  }
  return Storage(c10::intrusive_ptr<StorageImpl>::reclaim(static_cast<StorageImpl*>(th_pointer)));
}

${type_method_definitions}

static auto& registerer = globalATenDispatch()
  ${function_registrations};
}

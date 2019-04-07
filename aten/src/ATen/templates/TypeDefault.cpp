#include <ATen/TypeDefault.h>

// ${generated_comment}

#include <ATen/core/SparseTensorRef.h>
#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <c10/core/Scalar.h>
#include <ATen/core/SparseTensorRef.h>
#include <c10/core/Storage.h>
#include <ATen/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <ATen/DeviceGuard.h>
#include <ATen/SparseTensorUtils.h>

namespace at {

Tensor & TypeDefault::copy_(Tensor & self, const Tensor & src, bool non_blocking) const {
  Tensor b_src;
  if (is_sparse()) {
    b_src = src;
  } else {
    std::tie(b_src) = expand_inplace(self, src, "copy");
  }
  return s_copy_(self, b_src, non_blocking);
}

Tensor TypeDefault::copy(const Tensor & src, bool non_blocking, optional<Device> to_device) const {
  AT_CHECK(src.defined(), "attempt to copy an undefined tensor");
  Tensor r;
  if (is_sparse()) {
    r = at::empty({0}, this->options(to_device));
  } else {
    r = at::empty(src.sizes(), this->options(to_device));
  }
  r.copy_(src, non_blocking);
  return r;
}

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
  return at::globalContext().getNonVariableType(b,scalarType());
}
Type & TypeDefault::toScalarType(ScalarType s) const {
  return at::globalContext().getNonVariableType(backend(),s);
}
Tensor TypeDefault::tensorWithAllocator(IntArrayRef sizes, Allocator* allocator) const {
  return tensorWithAllocator(sizes, detail::defaultStrides(sizes), std::move(allocator));
}
Tensor TypeDefault::tensorWithAllocator(IntArrayRef sizes, IntArrayRef strides, Allocator* allocator) const {
  auto storage = storageWithAllocator(detail::computeStorageSize(sizes, strides), std::move(allocator));
  return at::empty({0}, options()).set_(storage, 0, sizes, strides);
}

Storage TypeDefault::storageWithAllocator(int64_t size, Allocator* allocator) const {
  // Potentially the storage might be marked as resizable too here
  return Storage(typeMeta(), size, allocator, /*resizable=*/false);
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

}

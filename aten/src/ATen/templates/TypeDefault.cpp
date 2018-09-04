#include "ATen/TypeDefault.h"

// ${generated_comment}

#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Scalar.h"
#include "ATen/core/SparseTensorRef.h"
#include "ATen/Storage.h"
#include "ATen/Tensor.h"
#include "ATen/core/TensorOptions.h"
#include "ATen/DeviceGuard.h"

namespace at {

Tensor & TypeDefault::copy_(Tensor & self, const Tensor & src, bool non_blocking) const {
  Tensor b_src;
  std::tie(b_src) = expand_inplace(self, src, "copy");
  return s_copy_(self, b_src, non_blocking);
}

Tensor TypeDefault::copy(const Tensor & src, bool non_blocking, optional<Device> to_device) const {
  DeviceGuard device_guard;
  if (to_device.has_value()) {
    device_guard.set_index(to_device.value().index());
  }
  AT_CHECK(src.defined(), "attempt to copy an undefined tensor");
  if (is_sparse()) {
    auto indices = src._indices();
    auto values = src._values();
    auto & this_dense = toBackend(is_cuda() ? Backend::CUDA : Backend::CPU);
    auto & this_dense_idx = this_dense.toScalarType(ScalarType::Long);
    auto indices_copy = this_dense_idx.copy(indices, non_blocking);
    auto values_copy = this_dense.copy(values, non_blocking);
    return _sparse_coo_tensor_unsafe(indices_copy, values_copy, src.sizes());
  } else {
    Tensor r = this->tensor(src.sizes());
    r.copy_(src, non_blocking);
    return r;
  }
}

Type & TypeDefault::toBackend(Backend b) const {
  return at::globalContext().getNonVariableType(b,scalarType());
}
Type & TypeDefault::toScalarType(ScalarType s) const {
  return at::globalContext().getNonVariableType(backend(),s);
}
static std::vector<int64_t> defaultStrides(IntList sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for(size_t i = sizes.size(); i > 0; --i) {
    strides[i-1] = stride;
    stride *= sizes[i-1];
  }
  return strides;
}
static int64_t computeStorageSize(IntList sizes, IntList strides) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  int64_t size = 1;
  for(size_t i = 0; i < sizes.size(); i++) {
    if(sizes[i] == 0) {
      return 0;
    }
    size += strides[i]*(sizes[i]-1);
  }
  return size;
}
Tensor TypeDefault::tensorFromBlob(void * data, IntList sizes, const std::function<void(void*)> & deleter) const {
  return tensorFromBlob(data, sizes, defaultStrides(sizes), deleter);
}
Tensor TypeDefault::tensorFromBlob(void * data, IntList sizes, IntList strides, const std::function<void(void*)> & deleter) const {
  auto storage = storageFromBlob(data, computeStorageSize(sizes, strides), deleter);
  return tensor(storage, 0, sizes, strides);
}
Tensor TypeDefault::tensorWithAllocator(IntList sizes, Allocator* allocator) const {
  return tensorWithAllocator(sizes, defaultStrides(sizes), std::move(allocator));
}
Tensor TypeDefault::tensorWithAllocator(IntList sizes, IntList strides, Allocator* allocator) const {
  auto storage = storageWithAllocator(computeStorageSize(sizes, strides), std::move(allocator));
  return tensor(storage, 0, sizes, strides);
}
Tensor TypeDefault::scalarTensor(Scalar s) const {
  return tensor({}).fill_(s);
}

${type_method_definitions}

}

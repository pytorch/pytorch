#include "ATen/Type.h"
#include "ATen/Tensor.h"
#include "ATen/Storage.h"
#include "ATen/Scalar.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/UndefinedType.h"

#include <iostream>
${type_headers}

namespace at {

void Type::registerAll(Context * context) {
  ${type_registrations}
  context->type_registry[static_cast<int>(Backend::Undefined)][static_cast<int>(ScalarType::Undefined)].reset(new UndefinedType(context));
}

Tensor & Type::copy_(Tensor & self, const Tensor & src, bool non_blocking) const {
  Tensor b_src;
  std::tie(b_src) = expand_inplace(self, src, "copy");
  return s_copy_(self, b_src, non_blocking);
}

Tensor Type::copy(const Tensor & src, bool non_blocking) const {
  AT_ASSERT(src.defined(), "attempt to copy an undefined tensor");
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

Type & Type::toBackend(Backend b) const {
  return context->getType(b,scalarType());
}
Type & Type::toScalarType(ScalarType s) const {
  return context->getType(backend(),s);
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
Tensor Type::tensorFromBlob(void * data, IntList sizes, const std::function<void(void*)> & deleter) const {
  return tensorFromBlob(data, sizes, defaultStrides(sizes), deleter);
}
Tensor Type::tensorFromBlob(void * data, IntList sizes, IntList strides, const std::function<void(void*)> & deleter) const {
  auto storage = storageFromBlob(data, computeStorageSize(sizes, strides), deleter);
  return tensor(*storage, 0, sizes, strides);
}
Tensor Type::tensorWithAllocator(IntList sizes, std::unique_ptr<Allocator> allocator) const {
  return tensorWithAllocator(sizes, defaultStrides(sizes), std::move(allocator));
}
Tensor Type::tensorWithAllocator(IntList sizes, IntList strides, std::unique_ptr<Allocator> allocator) const {
  auto storage = storageWithAllocator(computeStorageSize(sizes, strides), std::move(allocator));
  return tensor(*storage, 0, sizes, strides);
}
Tensor Type::scalarTensor(Scalar s) const {
  if(s.isBackedByTensor())
    return Tensor(s.t).toType(*this);
  return tensor({}).fill_(s);
}

bool Type::operator==(const Type& other) const {
  return this == &other;
}
bool Type::operator!=(const Type& other) const {
  return this != &other;
}

${type_method_definitions}

}

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

void Type::copy(const Tensor & src, Tensor & dst) const {
  Tensor b_src;
  std::tie(b_src) = expand_inplace(dst, src, "copy");
  s_copy(b_src, dst);
}

Tensor Type::copy(const Tensor & src) const {
  AT_ASSERT(src.defined(), "attempt to copy an undefined tensor");
  Tensor r = this->tensor(src.sizes());
  r.copy_(src);
  return r;
}

Type & Type::toBackend(Backend b) const {
  return context->getType(b,scalarType());
}
Type & Type::toScalarType(ScalarType s) const {
  return context->getType(backend(),s);
}

Tensor Type::tensorFromBlob(void * data, IntList sizes, const std::function<void(void*)> & deleter) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for(size_t i = sizes.size(); i > 0; --i) {
    strides[i-1] = stride;
    stride *= sizes[i-1];
  }
  return tensorFromBlob(data, sizes, strides, deleter);
}
Tensor Type::tensorFromBlob(void * data, IntList sizes, IntList strides, const std::function<void(void*)> & deleter) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  int64_t size = 1;
  for(size_t i = 0; i < sizes.size(); i++) {
    if(sizes[i] == 0) {
      size = 0;
      break;
    }
    size += strides[i]*(sizes[i]-1);
  }
  auto storage = storageFromBlob(data,size,deleter);
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

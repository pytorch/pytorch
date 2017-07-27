#include "ATen/Type.h"
#include "ATen/Tensor.h"
#include "ATen/Storage.h"
#include "ATen/Scalar.h"
#include "ATen/SparseTensorRef.h"

#include <iostream>
${type_headers}

namespace at {

void Type::registerAll(Context * context) {
  ${type_registrations}
}

Tensor Type::copy(const Tensor & src) {
  Tensor r = this->tensor();
  r.copy_(src);
  return r;
}

Type & Type::toBackend(Backend b) {
  return context->getType(b,scalarType());
}
Type & Type::toScalarType(ScalarType s) {
  return context->getType(backend(),s);
}

Tensor Type::tensorFromBlob(void * data, IntList sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for(size_t i = sizes.size(); i > 0; --i) {
    strides[i-1] = stride;
    stride *= sizes[i-1];
  }
  return tensorFromBlob(data, sizes, strides);
}
Tensor Type::tensorFromBlob(void * data, IntList sizes, IntList strides) {
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
  auto storage = storageFromBlob(data,size);
  return tensor(*storage, 0, sizes, strides);
}
Tensor Type::scalarTensor(Scalar s) {
  if(s.isBackedByTensor())
    return s.t.toType(*this);
  return tensor({}).fill_(s);
}

bool Type::operator==(const Type& other) const {
  return this->ID() == other.ID();
}

${type_method_definitions}

}

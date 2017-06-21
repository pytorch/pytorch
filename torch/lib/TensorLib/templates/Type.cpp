#include "TensorLib/Type.h"
#include "TensorLib/Tensor.h"

${type_headers}

namespace tlib {

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


${type_method_definitions}

}

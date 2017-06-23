#include "ATen/${Tensor}.h"
#include "ATen/HalfConvert.h"

//sketch:

/*

class Type {
  // ctor is configurable as well
  Tensor new(...) {}
  // all  methods and pure functions, virtually dispatched
  virtual Tensor add(Tensor a, Tensor b);

  // no Tensor arguments but virtually dispatched on type
  virtual Tensor linspace(int a, int b);
  //
}

// base class of derived tensors
class TensorImpl {
  Type * type_; //non-virtual so that we don't double indirect unnecessarily ...

  // not virtual - Type * handles the virtual dispatch
  Tensor add(Tensor b) {
    type_->add(self,b);
  }
}

// non-method style dispatch
Tensor add(Tensor a, Tensor b) {
  return a->type_->add(a,b);
}


}

*/

namespace at {

${Tensor}::${Tensor}(Context* context)
: ${Tensor}(context,${THTensor}_new(${state})) {}

${Tensor}::${Tensor}(Context* context, ${THTensor} * tensor)
: TensorImpl(&context->getType(Backend::${Backend},ScalarType::${ScalarName})),
  tensor(tensor),
  context(context) {}
${Tensor}::~${Tensor}() {
  ${THTensor}_free(${state,} tensor);
}

const char * ${Tensor}::toString() const {
  return "${Tensor}";
}

IntList ${Tensor}::sizes() {
  if(isScalar())
    return IntList();
  return IntList(reinterpret_cast<int64_t*>(tensor->size),tensor->nDimension);
}
IntList ${Tensor}::strides() {
  if(isScalar())
    return IntList();
  return IntList(reinterpret_cast<int64_t*>(tensor->stride),tensor->nDimension);
}
int64_t ${Tensor}::dim() {
  if(isScalar())
    return 0;
  return ${THTensor}_nDimension(${state,}tensor);
}
Scalar ${Tensor}::localScalar() {
  AT_ASSERT(isScalar(),"localScalar() called on Tensor with %d dims",sizes().size());
  return Scalar(${to_at_half}(${THTensor}_get1d(${state,}tensor, 0)));
}

const char * ${Tensor}::typeString() {
  return "${Type}";
}

}

#include "ATen/${Tensor}.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

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
void ${Tensor}::assign_(Scalar s) {
  AT_ASSERT(isScalar(),"assign_() called on Tensor with %d dims",sizes().size());
  ${THTensor}_set1d(${state,}tensor, 0,${to_th_half}(s.to${ScalarName}()));
}

const char * ${Tensor}::typeString() {
  return "${Type}";
}

}

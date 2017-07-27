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
  return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
}

int64_t ${Tensor}::dim() {
  if(isScalar())
    return 0;
  int64_t d = ${THTensor}_nDimension(${state,}tensor);
  // See Note [Undefined-dim versus 0-dim]
  if (d != 0)
    return d;
  return kUndefinedDimensions;
}

const char * ${Tensor}::typeString() {
  return "${Type}";
}
void * ${Tensor}::unsafeGetTH(bool retain) {
  if (retain)
      ${THTensor}_retain(${state,} tensor);
  return tensor;
}

${TensorDenseOrSparse}

}

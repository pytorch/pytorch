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
  return ${THTensor}_nDimension(${state,}tensor);
}

const char * ${Tensor}::typeString() {
  return "${Type}";
}
void * ${Tensor}::unsafeGetTH() {
  return tensor;
}

${TensorDenseOrSparse}

}

// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

// ${generated_comment}

#include "ATen/Config.h"
#include "ATen/${Tensor}.h"
#include "ATen/${Storage}.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

$extra_cuda_headers

namespace at {

namespace detail {
  ${Tensor}* new_${Tensor}() {
    return new ${Tensor}(${THTensor}_new(${state}));
  }
}

${Tensor}::${Tensor}(${THTensor} * tensor)
: TensorImpl(&globalContext().getType(Backend::${Backend},ScalarType::${ScalarName}), tensor)
{}

${Tensor}::~${Tensor}() {
  if (tensor) tensor->release();
}

const char * ${Tensor}::toString() const {
  return "${Tensor}";
}

IntList ${Tensor}::sizes() const {
  // NB: dim in tensor is not synchronized with THTensor, so it's
  // important to apply dim here
  return IntList(THTensor_getSizePtr(tensor), dim());
}

int64_t ${Tensor}::dim() const {
  if(isScalar())
    return 0;
  return tensor->dim();
}

const char * ${Tensor}::typeString() {
  return "${Type}";
}
void * ${Tensor}::unsafeGetTH(bool retain) {
  if (retain) {
    tensor->retain();
  }
  return tensor;
}

void ${Tensor}::release_resources() {
  tensor->release();
  tensor = nullptr;
}

${TensorDenseOrSparse}

}

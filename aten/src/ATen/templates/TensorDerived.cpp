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

${TensorDenseOrSparse}

}

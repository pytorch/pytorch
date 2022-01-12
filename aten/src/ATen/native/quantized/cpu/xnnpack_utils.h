#pragma once

#ifdef USE_XNNPACK
#include <ATen/native/xnnpack/Common.h>

using xnnpack_operator = at::native::xnnpack::Operator;

namespace at {
namespace native {
namespace xnnp_utils {

  at::Tensor reorder_weights_for_transpose_qconv(const at::Tensor&, int);

}  // xnnp_utils
}  // native
}  // at
#endif  // USE_XNNPACK

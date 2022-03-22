#pragma once

#ifdef USE_XNNPACK
#include <ATen/native/xnnpack/Common.h>

using xnnpack_operator = at::native::xnnpack::Operator;

namespace at {
namespace native {
namespace xnnp_utils {

std::vector<size_t> get_mem_format_aware_shape(const at::Tensor& in);

} // namespace xnnp_utils
} // namespace native
} // namespace at

#endif // USE_XNNPACK

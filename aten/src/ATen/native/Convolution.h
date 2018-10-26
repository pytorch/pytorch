#pragma once

#include <inttypes.h>

namespace at {
namespace native {
namespace detail {
enum class ConvolutionPaddingType : int64_t { ZEROS = 0, CIRCULAR = 1, END };
}
} // namespace native
} // namespace at

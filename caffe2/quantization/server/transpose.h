#pragma once

#include <cstdint>
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace fbgemm {

/**
 * Transpose 4xN matrix with unsigned 8-byte integers
 * TODO: move this to fbgemm after making transpose routine more general
 */
CAFFE2_API void transpose_4rows(int N, const std::uint8_t* src, std::uint8_t* dst);

} // namespace fbgemm

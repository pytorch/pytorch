#pragma once

#include <cstdint>
#include "c10/macros/Macros.h"

namespace fbgemm {

/**
 * Transpose 4xN matrix with unsigned 8-byte integers
 * TODO: move this to fbgemm after making transpose routine more general
 */
CAFFE2_API void
transpose_4rows(int N, const std::uint8_t* src, std::uint8_t* dst);

} // namespace fbgemm

#pragma once

#include <cstdint>

namespace fbgemm {

/**
 * Transpose 4xN matrix with unsigned 8-byte integers
 * TODO: move this to fbgemm after making transpose routine more general
 */
void transpose_4rows(int N, const std::uint8_t* src, std::uint8_t* dst);

} // namespace fbgemm

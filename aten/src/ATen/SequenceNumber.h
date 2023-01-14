#pragma once

#include <c10/macros/Export.h>
#include <cstdint>

namespace at::sequence_number {

TORCH_API uint64_t peek();
TORCH_API uint64_t get_and_increment();

} // namespace at::sequence_number

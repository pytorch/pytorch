#pragma once
#include <c10/macros/Export.h>
#include <cstdint>

namespace at::mps {

TORCH_API void sleep(int64_t cycles);

} // namespace at::mps

#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>

#include <c10/util/Exception.h>

#define INTRA_OP_PARALLEL

namespace at {
namespace internal {

TORCH_API void invoke_parallel(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const std::function<void(int64_t, int64_t)>& f);

} // namespace internal

} // namespace at

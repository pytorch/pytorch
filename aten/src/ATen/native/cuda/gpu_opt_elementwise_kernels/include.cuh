#pragma once

// GPU elementwise aggregation header. It owns the namespace boundary when
// consumed outside at::native and combines logging with dispatch entry points.
#include <c10/util/IntegerDivider.h>
#include "utils/log.h"

namespace at::native {
// CUDALoops.cuh defines these templates later, after its local helper
// declarations. Forward declarations preserve that definition order while
// allowing this aggregation header to remain the sole GPU include at top level.
template <int io_sizes>
constexpr auto elems_per_thread();

template <typename func_t>
constexpr auto calc_io_size();

#include "dispatch.cuh"
} // namespace at::native

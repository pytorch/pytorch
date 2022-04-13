#pragma once
#include <c10/util/Flags.h>

// TODO(whc) either deprecate this, or use it for all shape inference
C10_DECLARE_int(torch_lazy_ts_shape_cache_size);

// TODO(whc) unclear if this is useful, has only been tested as true
C10_DECLARE_bool(torch_lazy_ts_tensor_update_sync);

C10_DECLARE_bool(torch_lazy_ts_cuda);

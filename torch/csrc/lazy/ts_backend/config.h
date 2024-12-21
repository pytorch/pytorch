#pragma once
#include <c10/util/Flags.h>

// TODO(whc) unclear if this is useful, has only been tested as true
TORCH_DECLARE_bool(torch_lazy_ts_tensor_update_sync);

TORCH_DECLARE_bool(torch_lazy_ts_cuda);

#pragma once

#include <c10/macros/Export.h>

namespace torch::verbose {
TORCH_API int _mkl_set_verbose(int enable);
TORCH_API int _onednn_set_verbose(int level);
} // namespace torch::verbose

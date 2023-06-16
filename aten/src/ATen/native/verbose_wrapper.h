#ifndef VERBOSE_WRAPPER_H
#define VERBOSE_WRAPPER_H
#include <c10/macros/Export.h>

namespace torch {
namespace verbose {
TORCH_API int _mkl_set_verbose(int enable);
TORCH_API int _mkldnn_set_verbose(int level);
} // namespace verbose
} // namespace torch

#endif // VERBOSE_WRAPPER_H

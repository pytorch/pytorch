#ifndef VERBOSE_WRAPPER_H
#define VERBOSE_WRAPPER_H

namespace torch {
namespace verbose {
int _mkl_set_verbose(int enable);
int _mkldnn_set_verbose(int level);
} // namespace verbose
} // namespace torch

#endif // VERBOSE_WRAPPER_H

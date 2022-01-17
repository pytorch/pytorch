#pragma once

#include <ATen/Config.h>

// MKL Sparse is not currently supported on Windows
// See https://github.com/pytorch/pytorch/pull/50937#issuecomment-779272492
#if AT_MKL_ENABLED() && (!defined(_WIN32))
#define AT_USE_MKL_SPARSE() 1
#else
#define AT_USE_MKL_SPARSE() 0
#endif

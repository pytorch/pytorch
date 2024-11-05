#pragma once

#include <ATen/Config.h>

// MKL Sparse is not currently supported on Windows
// See https://github.com/pytorch/pytorch/issues/97352
#if AT_MKL_ENABLED()
#define AT_USE_MKL_SPARSE() 1
#else
#define AT_USE_MKL_SPARSE() 0
#endif

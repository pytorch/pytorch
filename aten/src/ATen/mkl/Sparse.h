#pragma once

#include <ATen/Config.h>

// MKL Sparse is not currently supported on Windows or Mac
#if AT_MKL_ENABLED() && (!defined(_MSC_VER) || !defined(__APPLE__) || !defined(__MACH__))
#define AT_USE_MKL_SPARSE() 1
#else
#define AT_USE_MKL_SPARSE() 0
#endif

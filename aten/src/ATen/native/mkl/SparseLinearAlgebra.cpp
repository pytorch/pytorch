#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKL_ENABLED()

namespace at { namespace native {

    
}}

#else  // AT_MKL_ENABLED

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

#include <mkl.h>
#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/Descriptors.h>
#include <ATen/mkl/Limits.h>

namespace at { namespace native {

}}
#endif

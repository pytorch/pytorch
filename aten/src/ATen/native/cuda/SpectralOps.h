#include "ATen/ATen.h"
#include "ATen/Config.h"
#include "ATen/Dispatch.h"
#include "ATen/Utils.h"
#include <ATen/optional.h>
#include "ATen/NativeFunctions.h"


namespace at { namespace native {

// Since ATen is now separated into CPU build and CUDA build, we need a way to
// call these functions only when CUDA is loaded. These implementations are
// called via CUDA hooks (at cuda/detail/CUDAHooks.cpp) from the actual native
// function counterparts (at native/SpectralOps.cpp), i.e.,
// _cufft_get_plan_cache_max_size, _cufft_set_plan_cache_max_size
// _cufft_get_plan_cache_size, and _cufft_clear_plan_cache.

int64_t __cufft_get_plan_cache_max_size_impl();

void __cufft_set_plan_cache_max_size_impl(int64_t max_size);

int64_t __cufft_get_plan_cache_size_impl();

void __cufft_clear_plan_cache_impl();

}} // namespace at::native

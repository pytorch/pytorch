#pragma once

// cub sort support for __nv_bfloat16 is added to cub 1.13 in
// https://github.com/NVIDIA/cub/pull/306 and according to
// https://github.com/NVIDIA/cub#releases, 1.13 is included in
// CUDA Toolkit 11.5
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11050
#define CUB_SUPPORTS_NV_BFLOAT16() 1
#else
#define CUB_SUPPORTS_NV_BFLOAT16() 0
#endif

// cub support for scan by key is scheduled to add to cub 1.15
// in https://github.com/NVIDIA/cub/pull/376, which version of
// CUDA toolkit will adopt cub 1.15 is not clear to me yet.
// All usage of such features are currently tested manually by
// cherry-picking that cub PR to local CUDA installation. This
// file will be updated when I know this information.
//                                               -- @zasdfgbnm
#if defined(CUDA_VERSION) && CUDA_VERSION >= 99999
#define CUB_SUPPORTS_SCAN_BY_KEY() 1
#else
#define CUB_SUPPORTS_SCAN_BY_KEY() 0
#endif

// cub support for scan by key is scheduled to add to cub 1.15
// in https://github.com/NVIDIA/cub/pull/305, which version of
// CUDA toolkit will adopt cub 1.15 is not clear to me yet.
// All usage of such features are currently tested manually by
// cherry-picking that cub PR to local CUDA installation. This
// file will be updated when I know this information.
//                                               -- @zasdfgbnm
#if defined(CUDA_VERSION) && CUDA_VERSION >= 99999
#define CUB_SUPPORTS_FUTURE_VALUE() 1
#else
#define CUB_SUPPORTS_FUTURE_VALUE() 0
#endif
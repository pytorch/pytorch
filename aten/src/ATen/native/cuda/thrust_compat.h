#pragma once

#if defined(USE_ROCM)
#include <thrust/distance.h>
#include <thrust/functional.h>
namespace TORCH_CUDA_STD_NS = ::thrust;
#else
#include <cuda/std/functional>
#include <cuda/std/iterator>
namespace TORCH_CUDA_STD_NS = ::cuda::std;
#endif

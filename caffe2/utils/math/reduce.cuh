#ifndef CAFFE2_UTILS_MATH_REDUCE_CUH_
#define CAFFE2_UTILS_MATH_REDUCE_CUH_

#include "caffe2/utils/cub_namespace.cuh"
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/common_gpu.h"

namespace caffe2 {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

template <typename T, int kBlockDimX, int kBlockDimY>
using BlockReduce2D = cub::
    BlockReduce<T, kBlockDimX, cub::BLOCK_REDUCE_WARP_REDUCTIONS, kBlockDimY>;

#define DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE_1(                       \
    size, Func, T, grid_dim, cuda_stream, ...)                                \
  do {                                                                        \
    if (size >= 128) {                                                        \
      Func<T, 1, 128>                                                         \
          <<<grid_dim, dim3(1, 128), 0, cuda_stream>>>(__VA_ARGS__);          \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
    } else if (size >= 64) {                                                  \
      Func<T, 2, 64><<<grid_dim, dim3(2, 64), 0, cuda_stream>>>(__VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
    } else if (size >= 32) {                                                  \
      Func<T, 4, 32><<<grid_dim, dim3(4, 32), 0, cuda_stream>>>(__VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
    } else {                                                                  \
      Func<T, 8, 16><<<grid_dim, dim3(8, 16), 0, cuda_stream>>>(__VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
    }                                                                         \
  } while (false)

#define DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE_2(              \
    size, Func, T1, T2, grid_dim, cuda_stream, ...)                  \
  do {                                                               \
    if (size >= 128) {                                               \
      Func<T1, T2, 1, 128>                                           \
          <<<grid_dim, dim3(1, 128), 0, cuda_stream>>>(__VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                \
    } else if (size >= 64) {                                         \
      Func<T1, T2, 2, 64>                                            \
          <<<grid_dim, dim3(2, 64), 0, cuda_stream>>>(__VA_ARGS__);  \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                \
    } else if (size >= 32) {                                         \
      Func<T1, T2, 4, 32>                                            \
          <<<grid_dim, dim3(4, 32), 0, cuda_stream>>>(__VA_ARGS__);  \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                \
    } else {                                                         \
      Func<T1, T2, 8, 16>                                            \
          <<<grid_dim, dim3(8, 16), 0, cuda_stream>>>(__VA_ARGS__);  \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                \
    }                                                                \
  } while (false)

} // namespace caffe2

#endif // CAFFE2_UTILS_MATH_REDUCE_CUH_

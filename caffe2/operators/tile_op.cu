#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/tile_op.h"

namespace caffe2 {
namespace {
__global__ void TileCopyKernel(
    int item_size,
    int outer_dim,
    int inner_dim,
    int tiles,
    const char* input_data,
    char* output_data) {
  CUDA_1D_KERNEL_LOOP(index, outer_dim * tiles) {
    int i = index / tiles;
    int t = index % tiles;
    const char* input_ptr = input_data + inner_dim * item_size * i;
    char* output_ptr = output_data + (i * tiles + t) * inner_dim * item_size;
    memcpy(output_ptr, input_ptr, inner_dim * item_size);
  }
}

template <typename T>
__global__ void TileGradientAxpyKernel(
    int outer_dim,
    int inner_dim,
    int tiles,
    const T* input_data,
    T* output_data) {
  typedef cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS> BlockReduce;

  for (int idx = blockIdx.x; idx < outer_dim * inner_dim; idx += gridDim.x) {
    int i = idx / inner_dim;
    int j = idx % inner_dim;
    T* output_ptr = output_data + inner_dim * i;

    T x = 0.0;
    for (int t = threadIdx.x; t < tiles; t += blockDim.x) {
      const T* input_ptr = input_data + (i * tiles + t) * inner_dim;
      x += input_ptr[j];
    }
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T totx = BlockReduce(temp_storage).Sum(x);
    if (threadIdx.x == 0) {
      output_ptr[j] = totx;
    }
    __syncthreads();
  }
}
} // namespace

template <>
void TileOp<CUDAContext>::DoTile(
    const TypeMeta& meta,
    int item_size,
    int outer_dim,
    int inner_dim,
    const char* input_data,
    char* output_data) {
  TileCopyKernel<<<
      std::min(outer_dim * tiles_, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      item_size, outer_dim, inner_dim, tiles_, input_data, output_data);
}

template <>
void TileGradientOp<float, CUDAContext>::DoTileGradient(
    const TypeMeta& meta,
    int item_size,
    int outer_dim,
    int inner_dim,
    const char* input_data,
    char* output_data) {
  TileGradientAxpyKernel<float><<<
      std::min(outer_dim * inner_dim, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      outer_dim,
      inner_dim,
      tiles_,
      reinterpret_cast<const float*>(input_data),
      reinterpret_cast<float*>(output_data));
}

REGISTER_CUDA_OPERATOR(Tile, TileOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(TileGradient, TileGradientOp<float, CUDAContext>);
} // namespace caffe2

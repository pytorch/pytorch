#include "THCUNN.h"
#include "THCTensor.hpp"
#include "common.h"

template <typename Dtype>
__global__ void MaxUnpoolForward(const int64_t nthreads, const Dtype* bottom_data, const int64_t* bottom_mask,
    const int64_t num, const int64_t channels, const int64_t iheight, const int64_t iwidth, const int64_t oheight, const int64_t owidth, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) { //index here indices the input pixels
    int64_t c = (index / iwidth / iheight) % channels;
    int64_t n = index / iwidth / iheight / channels;
    top_data += (n*channels + c)*oheight*owidth;
    int64_t maxind = bottom_mask[index] - TH_INDEX_BASE;

    top_data[maxind] = bottom_data[index];
  }
}

template <typename Dtype>
__global__ void MaxUnpoolBackward(const int64_t nthreads, const Dtype* top_diff, const int64_t* bottom_mask,
    const int64_t num, const int64_t channels, const int64_t iheight, const int64_t iwidth, const int64_t oheight, const int64_t owidth, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int64_t c = (index / iwidth / iheight) % channels;
    int64_t n = index / iwidth / iheight / channels;
    top_diff += (n*channels + c)*oheight*owidth;
    int64_t maxind = bottom_mask[index] - TH_INDEX_BASE;

    bottom_diff[index] = top_diff[maxind];
  }
}

#include "generic/SpatialMaxUnpooling.cu"
#include "THCGenerateFloatTypes.h"

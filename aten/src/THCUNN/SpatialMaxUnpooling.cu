#include "THCUNN.h"
#include "THCTensor.hpp"
#include "common.h"

template <typename Dtype>
__global__ void MaxUnpoolBackward(const int nthreads, const Dtype* top_diff, const int64_t* bottom_mask,
    const int num, const int channels, const int iheight, const int iwidth, const int oheight, const int owidth, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / iwidth / iheight) % channels;
    int n = index / iwidth / iheight / channels;
    top_diff += (n*channels + c)*oheight*owidth;
    int maxind = bottom_mask[index] - TH_INDEX_BASE;

    bottom_diff[index] = top_diff[maxind];
  }
}

#include "generic/SpatialMaxUnpooling.cu"
#include "THCGenerateFloatTypes.h"

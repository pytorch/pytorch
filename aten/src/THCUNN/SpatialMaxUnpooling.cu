#include "THCUNN.h"
#include "common.h"

template <typename Dtype>
__global__ void MaxUnpoolForward(const int nthreads, const Dtype* bottom_data, const int64_t* bottom_mask,
    const int num, const int channels, const int iheight, const int iwidth, const int oheight, const int owidth, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) { //index here indices the input pixels
    int c = (index / iwidth / iheight) % channels;
    int n = index / iwidth / iheight / channels;
    top_data += (n*channels + c)*oheight*owidth;
    int maxind = bottom_mask[index] - TH_INDEX_BASE;

    top_data[maxind] = bottom_data[index];
  }
}

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

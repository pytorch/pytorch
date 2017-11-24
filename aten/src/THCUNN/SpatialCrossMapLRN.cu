#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "common.h"

template <typename Dtype, typename Acctype>
__global__ void
#if __CUDA_ARCH__ >= 320
__launch_bounds__(CUDA_NUM_THREADS)
#endif
LRNFillScale(const int nthreads, const Dtype* const in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size,
    const Dtype k, Dtype* const scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const in_off = in + offset;
    Dtype* const scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Acctype accum_scale = Acctype(0);
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = ScalarConvert<Acctype, Dtype>::to(k + accum_scale * alpha_over_size);
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = ScalarConvert<Acctype, Dtype>::to(k + accum_scale * alpha_over_size);
      ++head;
    }
  }
}

template <typename Dtype>
__global__ void LRNComputeOutput(const int nthreads, const Dtype* in,
    const Dtype* scale, const Dtype negative_beta, Dtype* out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

template <typename Dtype, typename Acctype>
__global__ void LRNComputeDiff(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const top_data,
    const Dtype* const scale, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype negative_beta,
    const Dtype cache_ratio, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const bottom_off = bottom_data + offset;
    const Dtype* const top_off = top_data + offset;
    const Dtype* const scale_off = scale + offset;
    const Dtype* const top_diff_off = top_diff + offset;
    Dtype* const bottom_diff_off = bottom_diff + offset;
    int head = 0;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Acctype accum_ratio = Acctype(0);
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
          scale_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
          scale_off[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
            top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
          ScalarConvert<Acctype, Dtype>::to(top_diff_off[(head - post_pad) * step]
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio);
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
            top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
          ScalarConvert<Acctype, Dtype>::to(top_diff_off[(head - post_pad) * step]
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio);
      ++head;
    }
  }
}


#include "generic/SpatialCrossMapLRN.cu"
#include "THCGenerateFloatTypes.h"

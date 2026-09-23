// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#ifndef USE_ROCM
#include <ATen/native/cuda/MemoryAccess.cuh>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/upsample_trilinear3d_native.h>
#include <ATen/ops/upsample_trilinear3d_backward_native.h>
#endif

namespace at::native {
namespace {

__device__ __forceinline__ size_t
idx_3d(const size_t nc,
    const size_t depth,
    const size_t height,
    const size_t width,
    const size_t z,
    const size_t y,
    const size_t x) {
  return ((nc * depth + z) * height + y) * width + x;
}

template <typename accscalar_t>
__device__ __forceinline__ void compute_output_range(
    const int input_pos,
    const accscalar_t scale,
    const int output_size,
    const bool align_corners,
    int& min_output,
    int& max_output) {
  if (scale == static_cast<accscalar_t>(0)) {
    min_output = input_pos == 0 ? 0 : 1;
    max_output = input_pos == 0 ? output_size - 1 : 0;
    return;
  }

  accscalar_t lo;
  accscalar_t hi;
  if (align_corners) {
    lo = static_cast<accscalar_t>(input_pos - 1) / scale;
    hi = static_cast<accscalar_t>(input_pos + 1) / scale;
  } else {
    lo = (static_cast<accscalar_t>(input_pos) - static_cast<accscalar_t>(0.5)) /
        scale - static_cast<accscalar_t>(0.5);
    hi = (static_cast<accscalar_t>(input_pos) + static_cast<accscalar_t>(1.5)) /
        scale - static_cast<accscalar_t>(0.5);
  }

  min_output = max(0, static_cast<int>(ceil(lo)));
  max_output = min(output_size - 1, static_cast<int>(floor(hi)));
}

template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ void upsample_trilinear3d_backward_gather(
    const size_t index,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* __restrict__ idata,
    const scalar_t* __restrict__ odata) {
  size_t index_temp = index;
  const int input_w = index_temp % input_width;
  index_temp /= input_width;
  const int input_h = index_temp % input_height;
  index_temp /= input_height;
  const int input_d = index_temp % input_depth;
  const size_t nc_idx = index_temp / input_depth;

  int output_d_min;
  int output_d_max;
  int output_h_min;
  int output_h_max;
  int output_w_min;
  int output_w_max;
  compute_output_range(
      input_d, rdepth, output_depth, align_corners, output_d_min, output_d_max);
  compute_output_range(
      input_h, rheight, output_height, align_corners, output_h_min, output_h_max);
  compute_output_range(
      input_w, rwidth, output_width, align_corners, output_w_min, output_w_max);

  accscalar_t grad_sum = 0;
  const size_t output_plane = static_cast<size_t>(output_height) * output_width;

  for (int output_d_idx = output_d_min; output_d_idx <= output_d_max; ++output_d_idx) {
    const accscalar_t input_dr = area_pixel_compute_source_index<accscalar_t>(
        rdepth, output_d_idx, align_corners, /*cubic=*/false);
    const int input_d_base = static_cast<int>(input_dr);
    const int input_dp = (input_d_base < input_depth - 1) ? 1 : 0;
    const accscalar_t input_d_lambda = input_dr - input_d_base;
    const accscalar_t input_d0_lambda = static_cast<accscalar_t>(1) - input_d_lambda;

    for (int output_h_idx = output_h_min; output_h_idx <= output_h_max; ++output_h_idx) {
      const accscalar_t input_hr = area_pixel_compute_source_index<accscalar_t>(
          rheight, output_h_idx, align_corners, /*cubic=*/false);
      const int input_h_base = static_cast<int>(input_hr);
      const int input_hp = (input_h_base < input_height - 1) ? 1 : 0;
      const accscalar_t input_h_lambda = input_hr - input_h_base;
      const accscalar_t input_h0_lambda = static_cast<accscalar_t>(1) - input_h_lambda;

      for (int output_w_idx = output_w_min; output_w_idx <= output_w_max; ++output_w_idx) {
        const accscalar_t input_wr = area_pixel_compute_source_index<accscalar_t>(
            rwidth, output_w_idx, align_corners, /*cubic=*/false);
        const int input_w_base = static_cast<int>(input_wr);
        const int input_wp = (input_w_base < input_width - 1) ? 1 : 0;
        const accscalar_t input_w_lambda = input_wr - input_w_base;
        const accscalar_t input_w0_lambda = static_cast<accscalar_t>(1) - input_w_lambda;

        accscalar_t weight = 0;
        if (input_d == input_d_base && input_h == input_h_base && input_w == input_w_base) {
          weight += input_d0_lambda * input_h0_lambda * input_w0_lambda;
        }
        if (input_d == input_d_base && input_h == input_h_base &&
            input_w == input_w_base + input_wp) {
          weight += input_d0_lambda * input_h0_lambda * input_w_lambda;
        }
        if (input_d == input_d_base && input_h == input_h_base + input_hp &&
            input_w == input_w_base) {
          weight += input_d0_lambda * input_h_lambda * input_w0_lambda;
        }
        if (input_d == input_d_base && input_h == input_h_base + input_hp &&
            input_w == input_w_base + input_wp) {
          weight += input_d0_lambda * input_h_lambda * input_w_lambda;
        }
        if (input_d == input_d_base + input_dp && input_h == input_h_base &&
            input_w == input_w_base) {
          weight += input_d_lambda * input_h0_lambda * input_w0_lambda;
        }
        if (input_d == input_d_base + input_dp && input_h == input_h_base &&
            input_w == input_w_base + input_wp) {
          weight += input_d_lambda * input_h0_lambda * input_w_lambda;
        }
        if (input_d == input_d_base + input_dp && input_h == input_h_base + input_hp &&
            input_w == input_w_base) {
          weight += input_d_lambda * input_h_lambda * input_w0_lambda;
        }
        if (input_d == input_d_base + input_dp && input_h == input_h_base + input_hp &&
            input_w == input_w_base + input_wp) {
          weight += input_d_lambda * input_h_lambda * input_w_lambda;
        }

        if (weight > 0) {
          const size_t output_index =
              nc_idx * static_cast<size_t>(output_depth) * output_plane +
              static_cast<size_t>(output_d_idx) * output_plane +
              static_cast<size_t>(output_h_idx) * output_width + output_w_idx;
          grad_sum += weight * static_cast<accscalar_t>(odata[output_index]);
        }
      }
    }
  }

  idata[index] = static_cast<scalar_t>(grad_sum);
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_trilinear3d_backward_gather_out_frame(
    const size_t numel,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* __restrict__ idata,
    const scalar_t* __restrict__ odata) {
  for (size_t index = blockDim.x * blockIdx.x + threadIdx.x; index < numel;
       index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    upsample_trilinear3d_backward_gather<scalar_t, accscalar_t>(
        index,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        rdepth,
        rheight,
        rwidth,
        align_corners,
        idata,
        odata);
  }
}

#ifndef USE_ROCM
template <typename scalar_t, typename accscalar_t, int vec_size>
__device__ __forceinline__ void upsample_trilinear3d_backward_gather_ndhwc(
    const size_t index,
    const int channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* __restrict__ idata,
    const scalar_t* __restrict__ odata) {
  using vector_t = memory::aligned_vector<scalar_t, vec_size>;
  const int channel_vectors = channels / vec_size;
  size_t index_temp = index;
  const int c = (index_temp % channel_vectors) * vec_size;
  index_temp /= channel_vectors;
  const int input_w = index_temp % input_width;
  index_temp /= input_width;
  const int input_h = index_temp % input_height;
  index_temp /= input_height;
  const int input_d = index_temp % input_depth;
  const size_t n = index_temp / input_depth;

  int output_d_min;
  int output_d_max;
  int output_h_min;
  int output_h_max;
  int output_w_min;
  int output_w_max;
  compute_output_range(
      input_d, rdepth, output_depth, align_corners, output_d_min, output_d_max);
  compute_output_range(
      input_h, rheight, output_height, align_corners, output_h_min, output_h_max);
  compute_output_range(
      input_w, rwidth, output_width, align_corners, output_w_min, output_w_max);

  accscalar_t grad_sum[vec_size] = {};

  for (int output_d_idx = output_d_min; output_d_idx <= output_d_max; ++output_d_idx) {
    const accscalar_t input_dr = area_pixel_compute_source_index<accscalar_t>(
        rdepth, output_d_idx, align_corners, /*cubic=*/false);
    const int input_d_base = static_cast<int>(input_dr);
    const int input_dp = (input_d_base < input_depth - 1) ? 1 : 0;
    const accscalar_t input_d_lambda = input_dr - input_d_base;
    const accscalar_t input_d0_lambda = static_cast<accscalar_t>(1) - input_d_lambda;

    for (int output_h_idx = output_h_min; output_h_idx <= output_h_max; ++output_h_idx) {
      const accscalar_t input_hr = area_pixel_compute_source_index<accscalar_t>(
          rheight, output_h_idx, align_corners, /*cubic=*/false);
      const int input_h_base = static_cast<int>(input_hr);
      const int input_hp = (input_h_base < input_height - 1) ? 1 : 0;
      const accscalar_t input_h_lambda = input_hr - input_h_base;
      const accscalar_t input_h0_lambda = static_cast<accscalar_t>(1) - input_h_lambda;

      for (int output_w_idx = output_w_min; output_w_idx <= output_w_max; ++output_w_idx) {
        const accscalar_t input_wr = area_pixel_compute_source_index<accscalar_t>(
            rwidth, output_w_idx, align_corners, /*cubic=*/false);
        const int input_w_base = static_cast<int>(input_wr);
        const int input_wp = (input_w_base < input_width - 1) ? 1 : 0;
        const accscalar_t input_w_lambda = input_wr - input_w_base;
        const accscalar_t input_w0_lambda = static_cast<accscalar_t>(1) - input_w_lambda;

        accscalar_t weight = 0;
        if (input_d == input_d_base && input_h == input_h_base && input_w == input_w_base) {
          weight += input_d0_lambda * input_h0_lambda * input_w0_lambda;
        }
        if (input_d == input_d_base && input_h == input_h_base &&
            input_w == input_w_base + input_wp) {
          weight += input_d0_lambda * input_h0_lambda * input_w_lambda;
        }
        if (input_d == input_d_base && input_h == input_h_base + input_hp &&
            input_w == input_w_base) {
          weight += input_d0_lambda * input_h_lambda * input_w0_lambda;
        }
        if (input_d == input_d_base && input_h == input_h_base + input_hp &&
            input_w == input_w_base + input_wp) {
          weight += input_d0_lambda * input_h_lambda * input_w_lambda;
        }
        if (input_d == input_d_base + input_dp && input_h == input_h_base &&
            input_w == input_w_base) {
          weight += input_d_lambda * input_h0_lambda * input_w0_lambda;
        }
        if (input_d == input_d_base + input_dp && input_h == input_h_base &&
            input_w == input_w_base + input_wp) {
          weight += input_d_lambda * input_h0_lambda * input_w_lambda;
        }
        if (input_d == input_d_base + input_dp && input_h == input_h_base + input_hp &&
            input_w == input_w_base) {
          weight += input_d_lambda * input_h_lambda * input_w0_lambda;
        }
        if (input_d == input_d_base + input_dp && input_h == input_h_base + input_hp &&
            input_w == input_w_base + input_wp) {
          weight += input_d_lambda * input_h_lambda * input_w_lambda;
        }

        if (weight > 0) {
          const size_t output_index =
              ((((n * output_depth + output_d_idx) * output_height + output_h_idx) *
                output_width + output_w_idx) *
               channels) +
              c;
          const vector_t output =
              reinterpret_cast<const vector_t*>(odata)[output_index / vec_size];
#pragma unroll
          for (int lane = 0; lane < vec_size; ++lane) {
            grad_sum[lane] += weight * static_cast<accscalar_t>(output.val[lane]);
          }
        }
      }
    }
  }

  const size_t input_index =
      ((((n * input_depth + input_d) * input_height + input_h) * input_width +
        input_w) *
       channels) +
      c;
  vector_t input;
#pragma unroll
  for (int lane = 0; lane < vec_size; ++lane) {
    input.val[lane] = static_cast<scalar_t>(grad_sum[lane]);
  }
  reinterpret_cast<vector_t*>(idata)[input_index / vec_size] = input;
}

template <typename scalar_t, typename accscalar_t, int vec_size>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_trilinear3d_backward_gather_ndhwc_out_frame(
    const size_t numel,
    const int channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* __restrict__ idata,
    const scalar_t* __restrict__ odata) {
  for (size_t index = blockDim.x * blockIdx.x + threadIdx.x; index < numel;
       index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    upsample_trilinear3d_backward_gather_ndhwc<scalar_t, accscalar_t, vec_size>(
        index,
        channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        rdepth,
        rheight,
        rwidth,
        align_corners,
        idata,
        odata);
  }
}
#endif

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void upsample_trilinear3d_out_frame(
    const int n,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor64<const scalar_t, 5> idata,
    PackedTensorAccessor64<scalar_t, 5> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int depth1 = idata.size(2);
  const int height1 = idata.size(3);
  const int width1 = idata.size(4);
  const int depth2 = odata.size(2);
  const int height2 = odata.size(3);
  const int width2 = odata.size(4);

  if (index < n) {
    const int w2 = (index % (height2 * width2)) % width2; // 0:width2-1
    const int h2 = (index % (height2 * width2)) / width2; // 0:height2-1
    const int t2 = index / (height2 * width2); // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int t1 = t2;
      const int h1 = h2;
      const int w1 = w2;

      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[n][c][t1][h1][w1];
          odata[n][c][t2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const accscalar_t t1r = area_pixel_compute_source_index<accscalar_t>(
        rdepth, t2, align_corners, /*cubic=*/false);
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - t1;
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
    //
    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val = t0lambda *
                (h0lambda *
                     (w0lambda * idata[n][c][t1][h1][w1] +
                      w1lambda * idata[n][c][t1][h1][w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n][c][t1][h1 + h1p][w1] +
                      w1lambda * idata[n][c][t1][h1 + h1p][w1 + w1p])) +
            t1lambda *
                (h0lambda *
                     (w0lambda * idata[n][c][t1 + t1p][h1][w1] +
                      w1lambda * idata[n][c][t1 + t1p][h1][w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n][c][t1 + t1p][h1 + h1p][w1] +
                      w1lambda * idata[n][c][t1 + t1p][h1 + h1p][w1 + w1p]));
        odata[n][c][t2][h2][w2] = static_cast<scalar_t>(val);
      }
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void upsample_trilinear3d_backward_out_frame(
    const int num_kernels,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    PackedTensorAccessor64<scalar_t, 5> idata,
    const PackedTensorAccessor64<const scalar_t, 5> odata,
    scalar_t* idata_ptr) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int depth1 = idata.size(2);
  const int height1 = idata.size(3);
  const int width1 = idata.size(4);
  const int depth2 = odata.size(2);
  const int height2 = odata.size(3);
  const int width2 = odata.size(4);

  const size_t i_numel = batchsize * channels * depth1 * height1 * width1;

  if (index < num_kernels) {
    const int w2 = (index % (height2 * width2)) % width2; // 0:width2-1
    const int h2 = (index % (height2 * width2)) / width2; // 0:height2-1
    const int t2 = index / (height2 * width2); // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int t1 = t2;
      const int h1 = h2;
      const int w1 = w2;

      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = odata[n][c][t1][h1][w1];
          idata[n][c][t2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const accscalar_t t1r = area_pixel_compute_source_index<accscalar_t>(
        rdepth, t2, align_corners, /*cubic=*/false);
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - t1;
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
    //
    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t d2val = odata[n][c][t2][h2][w2];
        const size_t nc = n * channels + c;
        fastAtomicAdd(
          idata_ptr,
          idx_3d(nc, depth1, height1, width1, t1, h1, w1),
          i_numel,
          static_cast<scalar_t>(t0lambda * h0lambda * w0lambda * d2val),
          true);
        fastAtomicAdd(
          idata_ptr,
          idx_3d(nc, depth1, height1, width1, t1, h1, w1 + w1p),
          i_numel,
          static_cast<scalar_t>(t0lambda * h0lambda * w1lambda * d2val),
          true);
        fastAtomicAdd(
          idata_ptr,
          idx_3d(nc, depth1, height1, width1, t1, h1 + h1p, w1),
          i_numel,
          static_cast<scalar_t>(t0lambda * h1lambda * w0lambda * d2val),
          true);
        fastAtomicAdd(
          idata_ptr,
          idx_3d(nc, depth1, height1, width1, t1, h1 + h1p, w1 + w1p),
          i_numel,
          static_cast<scalar_t>(t0lambda * h1lambda * w1lambda * d2val),
          true);
        fastAtomicAdd(
          idata_ptr,
          idx_3d(nc, depth1, height1, width1, t1 + t1p, h1, w1),
          i_numel,
          static_cast<scalar_t>(t1lambda * h0lambda * w0lambda * d2val),
          true);
        fastAtomicAdd(
          idata_ptr,
          idx_3d(nc, depth1, height1, width1, t1 + t1p, h1, w1 + w1p),
          i_numel,
          static_cast<scalar_t>(t1lambda * h0lambda * w1lambda * d2val),
          true);
        fastAtomicAdd(
          idata_ptr,
          idx_3d(nc, depth1, height1, width1, t1 + t1p, h1 + h1p, w1),
          i_numel,
          static_cast<scalar_t>(t1lambda * h1lambda * w0lambda * d2val),
          true);
        fastAtomicAdd(
          idata_ptr,
          idx_3d(nc, depth1, height1, width1, t1 + t1p, h1 + h1p, w1 + w1p),
          i_numel,
          static_cast<scalar_t>(t1lambda * h1lambda * w1lambda * d2val),
          true);
      }
    }
  }
}

static void upsample_trilinear3d_out_cuda_template(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_trilinear3d_out_cuda", {input_arg, output_arg});

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int input_depth = input.size(2);
  int input_height = input.size(3);
  int input_width = input.size(4);

  const int num_kernels = output_depth * output_height * output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 512);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), "upsample_trilinear3d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor64<const scalar_t, 5>();
        auto odata = output.packed_accessor64<scalar_t, 5>();

        const accscalar_t rdepth = area_pixel_compute_scale<accscalar_t>(
            input_depth, output_depth, align_corners, scales_d);
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        upsample_trilinear3d_out_frame<scalar_t, accscalar_t>
            <<<ceil_div(num_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                rdepth,
                rheight,
                rwidth,
                align_corners,
                idata,
                odata);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

static void upsample_trilinear3d_backward_out_cuda_template(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input_, "grad_input_", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_trilinear3d_backward_out_cuda",
      {grad_output_arg, grad_input_arg});

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int input_depth = input_size[2];
  int input_height = input_size[3];
  int input_width = input_size[4];

  Tensor grad_output = grad_output_.contiguous();

  // A contiguous tensor is required for the kernel launch config
  Tensor grad_input = grad_input_.contiguous();

  // Numbers are added atomically to grad_input tensor from multiple threads,
  // so it has to be initialized to zero.
  grad_input.zero_();

  const int num_kernels = output_depth * output_height * output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      grad_output.scalar_type(),
      "upsample_trilinear3d_backward_out_frame",
      [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.packed_accessor64<scalar_t, 5>();
        auto odata = grad_output.packed_accessor64<const scalar_t, 5>();
        scalar_t* idata_ptr = grad_input.mutable_data_ptr<scalar_t>();

        const accscalar_t rdepth = area_pixel_compute_scale<accscalar_t>(
            input_depth, output_depth, align_corners, scales_d);
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        upsample_trilinear3d_backward_out_frame<scalar_t, accscalar_t>
            <<<ceil_div(num_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                rdepth,
                rheight,
                rwidth,
                align_corners,
                idata,
                odata,
                idata_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (!grad_input_.is_contiguous()) {
            grad_input_.copy_(grad_input);
        }
  });
}

static void upsample_trilinear3d_backward_out_cuda_template_deterministic(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input_, "grad_input_", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_trilinear3d_backward_out_cuda",
      {grad_output_arg, grad_input_arg});

  const int output_depth = output_size[0];
  const int output_height = output_size[1];
  const int output_width = output_size[2];
  const int nbatch = input_size[0];
  const int channels = input_size[1];
  const int input_depth = input_size[2];
  const int input_height = input_size[3];
  const int input_width = input_size[4];

  if (grad_input_.numel() == 0) {
    return;
  }

  if (grad_output_.sizes() == grad_input_.sizes()) {
    grad_input_.copy_(grad_output_);
    return;
  }

  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_output_.scalar_type(),
      "upsample_trilinear3d_backward_gather",
      [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        const accscalar_t rdepth = area_pixel_compute_scale<accscalar_t>(
            input_depth, output_depth, align_corners, scales_d);
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

#ifndef USE_ROCM
        if (grad_input_.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
          Tensor grad_output =
              grad_output_.contiguous(at::MemoryFormat::ChannelsLast3d);
          constexpr int vector_size = 16 / sizeof(scalar_t);
          const bool vectorized =
              vector_size > 1 && channels % vector_size == 0 &&
              memory::can_vectorize_up_to<scalar_t>(reinterpret_cast<const char*>(
                  grad_input_.const_data_ptr<scalar_t>())) >= vector_size &&
              memory::can_vectorize_up_to<scalar_t>(reinterpret_cast<const char*>(
                  grad_output.const_data_ptr<scalar_t>())) >= vector_size;
          const size_t num_kernels = static_cast<size_t>(nbatch) * channels *
              input_depth * input_height * input_width;
          const size_t work_items = vectorized
              ? num_kernels / vector_size
              : num_kernels;
          const size_t num_blocks = std::min(
              ceil_div(work_items, static_cast<size_t>(num_threads)),
              static_cast<size_t>(
                  at::cuda::getCurrentDeviceProperties()->maxGridSize[0]));
          if (vectorized) {
            upsample_trilinear3d_backward_gather_ndhwc_out_frame<
                scalar_t, accscalar_t, vector_size><<<
                num_blocks, num_threads, 0, stream>>>(
                work_items,
                channels,
                input_depth,
                input_height,
                input_width,
                output_depth,
                output_height,
                output_width,
                rdepth,
                rheight,
                rwidth,
                align_corners,
                grad_input_.mutable_data_ptr<scalar_t>(),
                grad_output.const_data_ptr<scalar_t>());
          } else {
            upsample_trilinear3d_backward_gather_ndhwc_out_frame<
                scalar_t, accscalar_t, 1><<<num_blocks, num_threads, 0, stream>>>(
                work_items,
                channels,
                input_depth,
                input_height,
                input_width,
                output_depth,
                output_height,
                output_width,
                rdepth,
                rheight,
                rwidth,
                align_corners,
                grad_input_.mutable_data_ptr<scalar_t>(),
                grad_output.const_data_ptr<scalar_t>());
          }
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          return;
        }
#endif

        Tensor grad_output = grad_output_.contiguous();
        Tensor grad_input = grad_input_.contiguous();
        const size_t num_kernels = static_cast<size_t>(nbatch) * channels *
            input_depth * input_height * input_width;
        const size_t num_blocks = std::min(
            ceil_div(num_kernels, static_cast<size_t>(num_threads)),
            static_cast<size_t>(
                at::cuda::getCurrentDeviceProperties()->maxGridSize[0]));
        upsample_trilinear3d_backward_gather_out_frame<scalar_t, accscalar_t>
            <<<num_blocks, num_threads, 0, stream>>>(
                num_kernels,
                input_depth,
                input_height,
                input_width,
                output_depth,
                output_height,
                output_width,
                rdepth,
                rheight,
                rwidth,
                align_corners,
                grad_input.mutable_data_ptr<scalar_t>(),
                grad_output.const_data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (!grad_input_.is_contiguous()) {
          grad_input_.copy_(grad_input);
        }
      });
}

} // namespace

TORCH_IMPL_FUNC(upsample_trilinear3d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output) {
  upsample_trilinear3d_out_cuda_template(output, input, output_size, align_corners, scales_d, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_trilinear3d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& grad_input) {
#ifdef USE_ROCM
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("upsample_trilinear3d_backward_out_cuda");
  upsample_trilinear3d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
#else
  if (globalContext().deterministicAlgorithms()) {
    upsample_trilinear3d_backward_out_cuda_template_deterministic(
        grad_input,
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_d,
        scales_h,
        scales_w);
  } else {
    // See Note [Writing Nondeterministic Operations]
    // Nondeterministic because of atomicAdd usage
    globalContext().alertNotDeterministic("upsample_trilinear3d_backward_out_cuda");
    upsample_trilinear3d_backward_out_cuda_template(
        grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
  }
#endif
}

} // namespace at::native

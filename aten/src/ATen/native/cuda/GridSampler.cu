#include "ATen/ATen.h"
#include "ATen/native/GridSampler.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/detail/TensorInfo.cuh"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include "ATen/cuda/detail/KernelUtils.h"

namespace at { namespace native {

using namespace at::cuda::detail;

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

namespace {
  static __forceinline__ __device__
  int clip_coordinates(int in, int clip_limit) {
    return ::min(clip_limit - 1, ::max(in, static_cast<int>(0)));
  }

  static __forceinline__ __device__
  bool within_bounds_2d(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
  }

  static __forceinline__ __device__
  bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
    return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
  }

  template<typename scalar_t>
  static __forceinline__ __device__
  void safe_add_2d(scalar_t *data, int h, int w,
                   int sH, int sW, int H, int W,
                   scalar_t delta) {
    if (within_bounds_2d(h, w, H, W)) {
      atomicAdd(data + h * sH + w * sW, delta);
    }
  }

  template<typename scalar_t>
  static __forceinline__ __device__
  void safe_add_3d(scalar_t *data, int d, int h, int w,
                   int sD, int sH, int sW, int D, int H, int W,
                   scalar_t delta) {
    if (within_bounds_3d(d, h, w, D, H, W)) {
      atomicAdd(data + d * sD + h * sH + w * sW, delta);
    }
  }

  template <typename scalar_t>
  __launch_bounds__(1024)
  __global__ void grid_sampler_2d_kernel(
      const int nthreads,
      TensorInfo<scalar_t, int> input,
      TensorInfo<scalar_t, int> grid,
      TensorInfo<scalar_t, int> output,
      const GridSamplerPadding padding_mode) {

    int C = input.sizes[1];
    int inp_H = input.sizes[2];
    int inp_W = input.sizes[3];
    int out_H = grid.sizes[1];
    int out_W = grid.sizes[2];
    int inp_sN = input.strides[0];
    int inp_sC = input.strides[1];
    int inp_sH = input.strides[2];
    int inp_sW = input.strides[3];
    int grid_sN = grid.strides[0];
    int grid_sH = grid.strides[1];
    int grid_sW = grid.strides[2];
    int grid_sCoor = grid.strides[3];
    int out_sN = output.strides[0];
    int out_sC = output.strides[1];
    int out_sH = output.strides[2];
    int out_sW = output.strides[3];

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int n = index / (out_H * out_W);
      const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];

      // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
      float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
      float iyf = ((iy + 1.f) / 2) * (inp_H - 1);

      ix = static_cast<scalar_t>(ixf);
      iy = static_cast<scalar_t>(iyf);

      // get NE, NW, SE, SW pixel values from (x, y)
      int ix_nw = static_cast<int>(::floor(ixf));
      int iy_nw = static_cast<int>(::floor(iyf));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      scalar_t nw = (ix_se - ix)    * (iy_se - iy);
      scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
      scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
      scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

      // calculate bilinear weighted pixel value and set output pixel
      if (padding_mode == GridSamplerPadding::Border) {
        // clip coordinates to image borders
        ix_nw = clip_coordinates(ix_nw, inp_W);
        iy_nw = clip_coordinates(iy_nw, inp_H);
        ix_ne = clip_coordinates(ix_ne, inp_W);
        iy_ne = clip_coordinates(iy_ne, inp_H);
        ix_sw = clip_coordinates(ix_sw, inp_W);
        iy_sw = clip_coordinates(iy_sw, inp_H);
        ix_se = clip_coordinates(ix_se, inp_W);
        iy_se = clip_coordinates(iy_se, inp_H);
      }

      auto inp_ptr_NC = input.data + n * inp_sN;
      auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        *out_ptr_NCHW = static_cast<scalar_t>(0);
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
        }
      }
    }
  }

  template <typename scalar_t>
  __launch_bounds__(1024)
  __global__ void grid_sampler_3d_kernel(
      const int nthreads,
      TensorInfo<scalar_t, int> input,
      TensorInfo<scalar_t, int> grid,
      TensorInfo<scalar_t, int> output,
      const GridSamplerPadding padding_mode) {

    int C = input.sizes[1];
    int inp_D = input.sizes[2];
    int inp_H = input.sizes[3];
    int inp_W = input.sizes[4];
    int out_D = grid.sizes[1];
    int out_H = grid.sizes[2];
    int out_W = grid.sizes[3];
    int inp_sN = input.strides[0];
    int inp_sC = input.strides[1];
    int inp_sD = input.strides[2];
    int inp_sH = input.strides[3];
    int inp_sW = input.strides[4];
    int grid_sN = grid.strides[0];
    int grid_sD = grid.strides[1];
    int grid_sH = grid.strides[2];
    int grid_sW = grid.strides[3];
    int grid_sCoor = grid.strides[4];
    int out_sN = output.strides[0];
    int out_sC = output.strides[1];
    int out_sD = output.strides[2];
    int out_sH = output.strides[3];
    int out_sW = output.strides[4];

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int d = (index / (out_H * out_W)) % out_D;
      const int n = index / (out_D * out_H * out_W);
      const int grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

      // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
      float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
      float iyf = ((iy + 1.f) / 2) * (inp_H - 1);
      float izf = ((iz + 1.f) / 2) * (inp_D - 1);

      ix = static_cast<scalar_t>(ixf);
      iy = static_cast<scalar_t>(iyf);
      iz = static_cast<scalar_t>(izf);

      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      int ix_tnw = static_cast<int>(::floor(ix));
      int iy_tnw = static_cast<int>(::floor(iy));
      int iz_tnw = static_cast<int>(::floor(iz));

      int ix_tne = ix_tnw + 1;
      int iy_tne = iy_tnw;
      int iz_tne = iz_tnw;

      int ix_tsw = ix_tnw;
      int iy_tsw = iy_tnw + 1;
      int iz_tsw = iz_tnw;

      int ix_tse = ix_tnw + 1;
      int iy_tse = iy_tnw + 1;
      int iz_tse = iz_tnw;

      int ix_bnw = ix_tnw;
      int iy_bnw = iy_tnw;
      int iz_bnw = iz_tnw + 1;

      int ix_bne = ix_tnw + 1;
      int iy_bne = iy_tnw;
      int iz_bne = iz_tnw + 1;

      int ix_bsw = ix_tnw;
      int iy_bsw = iy_tnw + 1;
      int iz_bsw = iz_tnw + 1;

      int ix_bse = ix_tnw + 1;
      int iy_bse = iy_tnw + 1;
      int iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
      scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
      scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
      scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
      scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
      scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
      scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
      scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

      if (padding_mode == GridSamplerPadding::Border) {
        // clip coordinates to image borders
        ix_tnw = clip_coordinates(ix_tnw, inp_W);
        iy_tnw = clip_coordinates(iy_tnw, inp_H);
        iz_tnw = clip_coordinates(iz_tnw, inp_D);
        ix_tne = clip_coordinates(ix_tne, inp_W);
        iy_tne = clip_coordinates(iy_tne, inp_H);
        iz_tne = clip_coordinates(iz_tne, inp_D);
        ix_tsw = clip_coordinates(ix_tsw, inp_W);
        iy_tsw = clip_coordinates(iy_tsw, inp_H);
        iz_tsw = clip_coordinates(iz_tsw, inp_D);
        ix_tse = clip_coordinates(ix_tse, inp_W);
        iy_tse = clip_coordinates(iy_tse, inp_H);
        iz_tse = clip_coordinates(iz_tse, inp_D);
        ix_bnw = clip_coordinates(ix_bnw, inp_W);
        iy_bnw = clip_coordinates(iy_bnw, inp_H);
        iz_bnw = clip_coordinates(iz_bnw, inp_D);
        ix_bne = clip_coordinates(ix_bne, inp_W);
        iy_bne = clip_coordinates(iy_bne, inp_H);
        iz_bne = clip_coordinates(iz_bne, inp_D);
        ix_bsw = clip_coordinates(ix_bsw, inp_W);
        iy_bsw = clip_coordinates(iy_bsw, inp_H);
        iz_bsw = clip_coordinates(iz_bsw, inp_D);
        ix_bse = clip_coordinates(ix_bse, inp_W);
        iy_bse = clip_coordinates(iy_bse, inp_H);
        iz_bse = clip_coordinates(iz_bse, inp_D);
      }

      auto inp_ptr_NC = input.data + n * inp_sN;
      auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
        // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
        // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
        // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
        *out_ptr_NCDHW = static_cast<scalar_t>(0);
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
        }
      }
    }
  }

  template <typename scalar_t>
  __launch_bounds__(1024)
  __global__ void grid_sampler_2d_backward_kernel(
      const int nthreads,
      TensorInfo<scalar_t, int> grad_output,
      TensorInfo<scalar_t, int> input,
      TensorInfo<scalar_t, int> grid,
      TensorInfo<scalar_t, int> grad_input,  // initialized to zeros
      TensorInfo<scalar_t, int> grad_grid,   // initialized to empty
      const GridSamplerPadding padding_mode) {

    int C = input.sizes[1];
    int inp_H = input.sizes[2];
    int inp_W = input.sizes[3];
    int out_H = grid.sizes[1];
    int out_W = grid.sizes[2];
    int inp_sN = input.strides[0];
    int inp_sC = input.strides[1];
    int inp_sH = input.strides[2];
    int inp_sW = input.strides[3];
    int grid_sN = grid.strides[0];
    int grid_sH = grid.strides[1];
    int grid_sW = grid.strides[2];
    int grid_sCoor = grid.strides[3];
    int gOut_sN = grad_output.strides[0];
    int gOut_sC = grad_output.strides[1];
    int gOut_sH = grad_output.strides[2];
    int gOut_sW = grad_output.strides[3];
    int gInp_sN = grad_input.strides[0];
    int gInp_sC = grad_input.strides[1];
    int gInp_sH = grad_input.strides[2];
    int gInp_sW = grad_input.strides[3];
    int gGrid_sW = grad_grid.strides[2];

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int n = index / (out_H * out_W);
      const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];

      // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
      float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
      float iyf = ((iy + 1.f) / 2) * (inp_H - 1);

      ix = static_cast<scalar_t>(ixf);
      iy = static_cast<scalar_t>(iyf);

      // get NE, NW, SE, SW pixel values from (x, y)
      int ix_nw = static_cast<int>(::floor(ixf));
      int iy_nw = static_cast<int>(::floor(iyf));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      scalar_t nw = (ix_se - ix)    * (iy_se - iy);
      scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
      scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
      scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

      int ix_nw_cl, iy_nw_cl, ix_ne_cl, iy_ne_cl, ix_sw_cl, iy_sw_cl, ix_se_cl, iy_se_cl;

      // calculate bilinear weighted pixel value and set output pixel
      if (padding_mode == GridSamplerPadding::Border) {
        // clip coordinates to image borders
        ix_nw_cl = clip_coordinates(ix_nw, inp_W);
        iy_nw_cl = clip_coordinates(iy_nw, inp_H);
        ix_ne_cl = clip_coordinates(ix_ne, inp_W);
        iy_ne_cl = clip_coordinates(iy_ne, inp_H);
        ix_sw_cl = clip_coordinates(ix_sw, inp_W);
        iy_sw_cl = clip_coordinates(iy_sw, inp_H);
        ix_se_cl = clip_coordinates(ix_se, inp_W);
        iy_se_cl = clip_coordinates(iy_se, inp_H);
      } else {
        ix_nw_cl = ix_nw;
        iy_nw_cl = iy_nw;
        ix_ne_cl = ix_ne;
        iy_ne_cl = iy_ne;
        ix_sw_cl = ix_sw;
        iy_sw_cl = iy_sw;
        ix_se_cl = ix_se;
        iy_se_cl = iy_se;
      }

      scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
      scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
      scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
      scalar_t *inp_ptr_NC = input.data + n * inp_sN;
      for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, gInp_ptr_NC += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
        scalar_t gOut = *gOut_ptr_NCHW;

        // calculate and set grad_input
        safe_add_2d(gInp_ptr_NC, iy_nw_cl, ix_nw_cl, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut);
        safe_add_2d(gInp_ptr_NC, iy_ne_cl, ix_ne_cl, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut);
        safe_add_2d(gInp_ptr_NC, iy_sw_cl, ix_sw_cl, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut);
        safe_add_2d(gInp_ptr_NC, iy_se_cl, ix_se_cl, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut);

        // calculate grad_grid
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_2d(iy_nw_cl, ix_nw_cl, inp_H, inp_W)) {
          scalar_t nw_val = inp_ptr_NC[iy_nw_cl * inp_sH + ix_nw_cl * inp_sW];
          gix -= nw_val * (iy_se - iy) * gOut;
          giy -= nw_val * (ix_se - ix) * gOut;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_2d(iy_ne_cl, ix_ne_cl, inp_H, inp_W)) {
          scalar_t ne_val = inp_ptr_NC[iy_ne_cl * inp_sH + ix_ne_cl * inp_sW];
          gix += ne_val * (iy_sw - iy) * gOut;
          giy -= ne_val * (ix - ix_sw) * gOut;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_2d(iy_sw_cl, ix_sw_cl, inp_H, inp_W)) {
          scalar_t sw_val = inp_ptr_NC[iy_sw_cl * inp_sH + ix_sw_cl * inp_sW];
          gix -= sw_val * (iy - iy_ne) * gOut;
          giy += sw_val * (ix_ne - ix) * gOut;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_2d(iy_se_cl, ix_se_cl, inp_H, inp_W)) {
          scalar_t se_val = inp_ptr_NC[iy_se_cl * inp_sH + ix_se_cl * inp_sW];
          gix += se_val * (iy - iy_nw) * gOut;
          giy += se_val * (ix - ix_nw) * gOut;
        }
      }

      // un-normalize grad_grid values back to [-1, 1] constraints
      gix = gix * (inp_W - 1.f) / 2;
      giy = giy * (inp_H - 1.f) / 2;

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with gGrid_sW to diectly compute gGrid_ptr_NHW
      //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
      scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
      gGrid_ptr_NHW[0] = gix;
      gGrid_ptr_NHW[1] = giy;
    }
  }

  template <typename scalar_t>
  __launch_bounds__(1024)
  __global__ void grid_sampler_3d_backward_kernel(
      const int nthreads,
      TensorInfo<scalar_t, int> grad_output,
      TensorInfo<scalar_t, int> input,
      TensorInfo<scalar_t, int> grid,
      TensorInfo<scalar_t, int> grad_input,  // initialized to zeros
      TensorInfo<scalar_t, int> grad_grid,   // initialized to empty
      const GridSamplerPadding padding_mode) {

    int C = input.sizes[1];
    int inp_D = input.sizes[2];
    int inp_H = input.sizes[3];
    int inp_W = input.sizes[4];
    int out_D = grid.sizes[1];
    int out_H = grid.sizes[2];
    int out_W = grid.sizes[3];
    int inp_sN = input.strides[0];
    int inp_sC = input.strides[1];
    int inp_sD = input.strides[2];
    int inp_sH = input.strides[3];
    int inp_sW = input.strides[4];
    int grid_sN = grid.strides[0];
    int grid_sD = grid.strides[1];
    int grid_sH = grid.strides[2];
    int grid_sW = grid.strides[3];
    int grid_sCoor = grid.strides[4];
    int gOut_sN = grad_output.strides[0];
    int gOut_sC = grad_output.strides[1];
    int gOut_sD = grad_output.strides[2];
    int gOut_sH = grad_output.strides[3];
    int gOut_sW = grad_output.strides[4];
    int gInp_sN = grad_input.strides[0];
    int gInp_sC = grad_input.strides[1];
    int gInp_sD = grad_input.strides[2];
    int gInp_sH = grad_input.strides[3];
    int gInp_sW = grad_input.strides[4];
    int gGrid_sW = grad_grid.strides[3];

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int d = (index / (out_H * out_W)) % out_D;
      const int n = index / (out_D * out_H * out_W);
      const int grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

      // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
      float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
      float iyf = ((iy + 1.f) / 2) * (inp_H - 1);
      float izf = ((iz + 1.f) / 2) * (inp_D - 1);

      ix = static_cast<scalar_t>(ixf);
      iy = static_cast<scalar_t>(iyf);
      iz = static_cast<scalar_t>(izf);

      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      int ix_tnw = static_cast<int>(::floor(ix));
      int iy_tnw = static_cast<int>(::floor(iy));
      int iz_tnw = static_cast<int>(::floor(iz));

      int ix_tne = ix_tnw + 1;
      int iy_tne = iy_tnw;
      int iz_tne = iz_tnw;

      int ix_tsw = ix_tnw;
      int iy_tsw = iy_tnw + 1;
      int iz_tsw = iz_tnw;

      int ix_tse = ix_tnw + 1;
      int iy_tse = iy_tnw + 1;
      int iz_tse = iz_tnw;

      int ix_bnw = ix_tnw;
      int iy_bnw = iy_tnw;
      int iz_bnw = iz_tnw + 1;

      int ix_bne = ix_tnw + 1;
      int iy_bne = iy_tnw;
      int iz_bne = iz_tnw + 1;

      int ix_bsw = ix_tnw;
      int iy_bsw = iy_tnw + 1;
      int iz_bsw = iz_tnw + 1;

      int ix_bse = ix_tnw + 1;
      int iy_bse = iy_tnw + 1;
      int iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
      scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
      scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
      scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
      scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
      scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
      scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
      scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

      int ix_tnw_cl, iy_tnw_cl, iz_tnw_cl, ix_tne_cl, iy_tne_cl, iz_tne_cl;
      int ix_tsw_cl, iy_tsw_cl, iz_tsw_cl, ix_tse_cl, iy_tse_cl, iz_tse_cl;
      int ix_bnw_cl, iy_bnw_cl, iz_bnw_cl, ix_bne_cl, iy_bne_cl, iz_bne_cl;
      int ix_bsw_cl, iy_bsw_cl, iz_bsw_cl, ix_bse_cl, iy_bse_cl, iz_bse_cl;

      if (padding_mode == GridSamplerPadding::Border) {
        // clip coordinates to image borders
        ix_tnw_cl = clip_coordinates(ix_tnw, inp_W);
        iy_tnw_cl = clip_coordinates(iy_tnw, inp_H);
        iz_tnw_cl = clip_coordinates(iz_tnw, inp_D);
        ix_tne_cl = clip_coordinates(ix_tne, inp_W);
        iy_tne_cl = clip_coordinates(iy_tne, inp_H);
        iz_tne_cl = clip_coordinates(iz_tne, inp_D);
        ix_tsw_cl = clip_coordinates(ix_tsw, inp_W);
        iy_tsw_cl = clip_coordinates(iy_tsw, inp_H);
        iz_tsw_cl = clip_coordinates(iz_tsw, inp_D);
        ix_tse_cl = clip_coordinates(ix_tse, inp_W);
        iy_tse_cl = clip_coordinates(iy_tse, inp_H);
        iz_tse_cl = clip_coordinates(iz_tse, inp_D);
        ix_bnw_cl = clip_coordinates(ix_bnw, inp_W);
        iy_bnw_cl = clip_coordinates(iy_bnw, inp_H);
        iz_bnw_cl = clip_coordinates(iz_bnw, inp_D);
        ix_bne_cl = clip_coordinates(ix_bne, inp_W);
        iy_bne_cl = clip_coordinates(iy_bne, inp_H);
        iz_bne_cl = clip_coordinates(iz_bne, inp_D);
        ix_bsw_cl = clip_coordinates(ix_bsw, inp_W);
        iy_bsw_cl = clip_coordinates(iy_bsw, inp_H);
        iz_bsw_cl = clip_coordinates(iz_bsw, inp_D);
        ix_bse_cl = clip_coordinates(ix_bse, inp_W);
        iy_bse_cl = clip_coordinates(iy_bse, inp_H);
        iz_bse_cl = clip_coordinates(iz_bse, inp_D);
      } else {
        ix_tnw_cl = ix_tnw;
        iy_tnw_cl = iy_tnw;
        iz_tnw_cl = iz_tnw;
        ix_tne_cl = ix_tne;
        iy_tne_cl = iy_tne;
        iz_tne_cl = iz_tne;
        ix_tsw_cl = ix_tsw;
        iy_tsw_cl = iy_tsw;
        iz_tsw_cl = iz_tsw;
        ix_tse_cl = ix_tse;
        iy_tse_cl = iy_tse;
        iz_tse_cl = iz_tse;
        ix_bnw_cl = ix_bnw;
        iy_bnw_cl = iy_bnw;
        iz_bnw_cl = iz_bnw;
        ix_bne_cl = ix_bne;
        iy_bne_cl = iy_bne;
        iz_bne_cl = iz_bne;
        ix_bsw_cl = ix_bsw;
        iy_bsw_cl = iy_bsw;
        iz_bsw_cl = iz_bsw;
        ix_bse_cl = ix_bse;
        iy_bse_cl = iy_bse;
        iz_bse_cl = iz_bse;
      }

      scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0);
      scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
      scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
      scalar_t *inp_ptr_NC = input.data + n * inp_sN;
      // calculate bilinear weighted pixel value and set output pixel
      for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
        scalar_t gOut = *gOut_ptr_NCDHW;

        // calculate and set grad_input
        safe_add_3d(gInp_ptr_NC, iz_tnw_cl, iy_tnw_cl, ix_tnw_cl, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut);
        safe_add_3d(gInp_ptr_NC, iz_tne_cl, iy_tne_cl, ix_tne_cl, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut);
        safe_add_3d(gInp_ptr_NC, iz_tsw_cl, iy_tsw_cl, ix_tsw_cl, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut);
        safe_add_3d(gInp_ptr_NC, iz_tse_cl, iy_tse_cl, ix_tse_cl, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut);
        safe_add_3d(gInp_ptr_NC, iz_bnw_cl, iy_bnw_cl, ix_bnw_cl, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut);
        safe_add_3d(gInp_ptr_NC, iz_bne_cl, iy_bne_cl, ix_bne_cl, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut);
        safe_add_3d(gInp_ptr_NC, iz_bsw_cl, iy_bsw_cl, ix_bsw_cl, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut);
        safe_add_3d(gInp_ptr_NC, iz_bse_cl, iy_bse_cl, ix_bse_cl, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut);

        // calculate grad_grid
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_tnw_cl, iy_tnw_cl, ix_tnw_cl, inp_D, inp_H, inp_W)) {
          scalar_t tnw_val = inp_ptr_NC[iz_tnw_cl * inp_sD + iy_tnw_cl * inp_sH + ix_tnw_cl * inp_sW];
          gix -= tnw_val * (iy_bse - iy)    * (iz_bse - iz)    * gOut;
          giy -= tnw_val * (ix_bse - ix)    * (iz_bse - iz)    * gOut;
          giz -= tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * gOut;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_tne_cl, iy_tne_cl, ix_tne_cl, inp_D, inp_H, inp_W)) {
          scalar_t tne_val = inp_ptr_NC[iz_tne_cl * inp_sD + iy_tne_cl * inp_sH + ix_tne_cl * inp_sW];
          gix += tne_val * (iy_bsw - iy)    * (iz_bsw - iz)    * gOut;
          giy -= tne_val * (ix    - ix_bsw) * (iz_bsw - iz)    * gOut;
          giz -= tne_val * (ix    - ix_bsw) * (iy_bsw - iy)    * gOut;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_tsw_cl, iy_tsw_cl, ix_tsw_cl, inp_D, inp_H, inp_W)) {
          scalar_t tsw_val = inp_ptr_NC[iz_tsw_cl * inp_sD + iy_tsw_cl * inp_sH + ix_tsw_cl * inp_sW];
          gix -= tsw_val * (iy - iy_bne)    * (iz_bne - iz)    * gOut;
          giy += tsw_val * (ix_bne - ix)    * (iz_bne - iz)    * gOut;
          giz -= tsw_val * (ix_bne - ix)    * (iy    - iy_bne) * gOut;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_tse_cl, iy_tse_cl, ix_tse_cl, inp_D, inp_H, inp_W)) {
          scalar_t tse_val = inp_ptr_NC[iz_tse_cl * inp_sD + iy_tse_cl * inp_sH + ix_tse_cl * inp_sW];
          gix += tse_val * (iy - iy_bnw)    * (iz_bnw - iz)    * gOut;
          giy += tse_val * (ix    - ix_bnw) * (iz_bnw - iz)    * gOut;
          giz -= tse_val * (ix    - ix_bnw) * (iy    - iy_bnw) * gOut;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_bnw_cl, iy_bnw_cl, ix_bnw_cl, inp_D, inp_H, inp_W)) {
          scalar_t bnw_val = inp_ptr_NC[iz_bnw_cl * inp_sD + iy_bnw_cl * inp_sH + ix_bnw_cl * inp_sW];
          gix -= bnw_val * (iy_tse - iy)    * (iz - iz_tse)    * gOut;
          giy -= bnw_val * (ix_tse - ix)    * (iz - iz_tse)    * gOut;
          giz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * gOut;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_bne_cl, iy_bne_cl, ix_bne_cl, inp_D, inp_H, inp_W)) {
          scalar_t bne_val = inp_ptr_NC[iz_bne_cl * inp_sD + iy_bne_cl * inp_sH + ix_bne_cl * inp_sW];
          gix += bne_val * (iy_tsw - iy)    * (iz - iz_tsw)    * gOut;
          giy -= bne_val * (ix    - ix_tsw) * (iz - iz_tsw)    * gOut;
          giz += bne_val * (ix    - ix_tsw) * (iy_tsw - iy)    * gOut;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_bsw_cl, iy_bsw_cl, ix_bsw_cl, inp_D, inp_H, inp_W)) {
          scalar_t bsw_val = inp_ptr_NC[iz_bsw_cl * inp_sD + iy_bsw_cl * inp_sH + ix_bsw_cl * inp_sW];
          gix -= bsw_val * (iy - iy_tne)    * (iz - iz_tne)    * gOut;
          giy += bsw_val * (ix_tne - ix)    * (iz - iz_tne)    * gOut;
          giz += bsw_val * (ix_tne - ix)    * (iy    - iy_tne) * gOut;
        }
        if (padding_mode != GridSamplerPadding::Zeros || within_bounds_3d(iz_bse_cl, iy_bse_cl, ix_bse_cl, inp_D, inp_H, inp_W)) {
          scalar_t bse_val = inp_ptr_NC[iz_bse_cl * inp_sD + iy_bse_cl * inp_sH + ix_bse_cl * inp_sW];
          gix += bse_val * (iy - iy_tnw)    * (iz - iz_tnw)    * gOut;
          giy += bse_val * (ix    - ix_tnw) * (iz - iz_tnw)    * gOut;
          giz += bse_val * (ix    - ix_tnw) * (iy    - iy_tnw) * gOut;
        }
      }

      // un-normalize grad_grid values back to [-1, 1] constraints
      gix = gix * (inp_W - 1) / 2;
      giy = giy * (inp_H - 1) / 2;
      giz = giz * (inp_D - 1) / 2;

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with gGrid_sW to diectly compute gGrid_ptr_NDHW
      //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
      scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
      gGrid_ptr_NDHW[0] = gix;
      gGrid_ptr_NDHW[1] = giy;
      gGrid_ptr_NDHW[2] = giz;
    }
  }
}  // namespace

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode) {
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto output = at::empty({N, input.size(1), H, W}, input.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "grid_sampler_2d_cuda", [&] {
    int count = static_cast<int>(N * H * W);
    grid_sampler_2d_kernel<scalar_t>
      <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        count,
        getTensorInfo<scalar_t, int>(input),
        getTensorInfo<scalar_t, int>(grid),
        getTensorInfo<scalar_t, int>(output),
        static_cast<GridSamplerPadding>(padding_mode));
  });
  return output;
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_3d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode) {
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  auto output = at::empty({N, input.size(1), D, H, W}, input.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "grid_sampler_2d_cuda", [&] {
    int count = static_cast<int>(N * D * H * W);
    grid_sampler_3d_kernel<scalar_t>
      <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        count,
        getTensorInfo<scalar_t, int>(input),
        getTensorInfo<scalar_t, int>(grid),
        getTensorInfo<scalar_t, int>(output),
        static_cast<GridSamplerPadding>(padding_mode));
  });
  return output;
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                              int64_t interpolation_mode, int64_t padding_mode) {
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto grad_input = at::zeros_like(input);
  auto grad_grid = at::empty_like(grid);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "grid_sampler_2d_backward_cuda", [&] {
    int count = static_cast<int>(N * H * W);
    grid_sampler_2d_backward_kernel<scalar_t>
      <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        count,
        getTensorInfo<scalar_t, int>(grad_output),
        getTensorInfo<scalar_t, int>(input),
        getTensorInfo<scalar_t, int>(grid),
        getTensorInfo<scalar_t, int>(grad_input),
        getTensorInfo<scalar_t, int>(grad_grid),
        static_cast<GridSamplerPadding>(padding_mode));
  });
  return std::make_tuple(grad_input, grad_grid);
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                              int64_t interpolation_mode, int64_t padding_mode) {
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  auto grad_input = at::zeros_like(input);
  auto grad_grid = at::empty_like(grid);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "grid_sampler_3d_backward_cuda", [&] {
    int count = static_cast<int>(N * D * H * W);
    grid_sampler_3d_backward_kernel<scalar_t>
      <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        count,
        getTensorInfo<scalar_t, int>(grad_output),
        getTensorInfo<scalar_t, int>(input),
        getTensorInfo<scalar_t, int>(grid),
        getTensorInfo<scalar_t, int>(grad_input),
        getTensorInfo<scalar_t, int>(grad_grid),
        static_cast<GridSamplerPadding>(padding_mode));
  });
  return std::make_tuple(grad_input, grad_grid);
}

}}  // namespace at::native

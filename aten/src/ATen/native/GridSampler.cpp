#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/detail/CUDAHooksInterface.h"
#include "ATen/native/GridSampler.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace at { namespace native {

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

namespace {
  static inline int64_t clip_coordinates(int64_t in, int64_t clip_limit) {
    return std::min(clip_limit - 1, std::max(in, static_cast<int64_t>(0)));
  }

  static inline bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W) {
    return h >= 0 && h < H && w >= 0 && w < W;
  }

  static inline bool within_bounds_3d(int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W) {
    return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
  }

  template<typename scalar_t>
  static inline void safe_add_2d(scalar_t *data, int64_t h, int64_t w,
                                 int64_t sH, int64_t sW, int64_t H, int64_t W,
                                 scalar_t delta) {
    if (within_bounds_2d(h, w, H, W)) {
      data[h * sH + w * sW] += delta;
    }
  }

  template<typename scalar_t>
  static inline void safe_add_3d(scalar_t *data, int64_t d, int64_t h, int64_t w,
                                 int64_t sD, int64_t sH, int64_t sW,
                                 int64_t D, int64_t H, int64_t W,
                                 scalar_t delta) {
    if (within_bounds_3d(d, h, w, D, H, W)) {
      data[d * sD + h * sH + w * sW] += delta;
    }
  }

  template<typename scalar_t>
  Tensor grid_sampler2d_cpu_impl(const Tensor& input, const Tensor& grid,
                                 GridSamplerInterpolation interpolation_mode,
                                 GridSamplerPadding padding_mode) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_H = input.size(2);
    int64_t inp_W = input.size(3);
    int64_t out_H = grid.size(1);
    int64_t out_W = grid.size(2);
    auto output = at::empty({N, C, out_H, out_W}, input.options());
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sH = input.stride(2);
    int64_t inp_sW = input.stride(3);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sH = grid.stride(1);
    int64_t grid_sW = grid.stride(2);
    int64_t grid_sCoor = grid.stride(3);
    int64_t out_sN = output.stride(0);
    int64_t out_sC = output.stride(1);
    int64_t out_sH = output.stride(2);
    int64_t out_sW = output.stride(3);
    scalar_t *inp_ptr = input.data<scalar_t>();
    scalar_t *out_ptr = output.data<scalar_t>();
    scalar_t *grid_ptr = grid.data<scalar_t>();
    // loop over each output pixel
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t n = 0; n < N; ++n) {
      scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
      scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
      for (int64_t h = 0; h < out_H; ++h) {
        for (int64_t w = 0; w < out_W; ++w) {
          // get the corresponding input x, y co-ordinates from grid
          scalar_t ix = grid_ptr_N[h * grid_sH + w * grid_sW];
          scalar_t iy = grid_ptr_N[h * grid_sH + w * grid_sW + grid_sCoor];

          // normalize ix, iy from [-1, 1] to [0, inp_W-1] & [0, inp_H-1]
          ix = ((ix + 1) / 2) * (inp_W - 1);
          iy = ((iy + 1) / 2) * (inp_H - 1);

          // get NE, NW, SE, SW pixel values from (x, y)
          int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
          int64_t iy_nw = static_cast<int64_t>(std::floor(iy));
          int64_t ix_ne = ix_nw + 1;
          int64_t iy_ne = iy_nw;
          int64_t ix_sw = ix_nw;
          int64_t iy_sw = iy_nw + 1;
          int64_t ix_se = ix_nw + 1;
          int64_t iy_se = iy_nw + 1;

          // get surfaces to each neighbor:
          scalar_t nw = (ix_se - ix)    * (iy_se - iy);
          scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
          scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
          scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

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

          // calculate bilinear weighted pixel value and set output pixel
          scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
          scalar_t *inp_ptr_NC = inp_ptr_N;
          for (int c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
            //   (c, iy_nw, ix_nw) * nw + (c, iy_ne, ix_ne) * ne
            // + (c, iy_sw, ix_sw) * sw + (c, iy_se, ix_se) * se
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
    }
    return output;
  }

  template<typename scalar_t>
  Tensor grid_sampler3d_cpu_impl(const Tensor& input, const Tensor& grid,
                                 GridSamplerInterpolation interpolation_mode,
                                 GridSamplerPadding padding_mode) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_D = input.size(2);
    int64_t inp_H = input.size(3);
    int64_t inp_W = input.size(4);
    int64_t out_D = grid.size(1);
    int64_t out_H = grid.size(2);
    int64_t out_W = grid.size(3);
    auto output = at::empty({N, C, out_D, out_H, out_W}, input.options());
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sD = input.stride(2);
    int64_t inp_sH = input.stride(3);
    int64_t inp_sW = input.stride(4);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sD = grid.stride(1);
    int64_t grid_sH = grid.stride(2);
    int64_t grid_sW = grid.stride(3);
    int64_t grid_sCoor = grid.stride(4);
    int64_t out_sN = output.stride(0);
    int64_t out_sC = output.stride(1);
    int64_t out_sD = output.stride(2);
    int64_t out_sH = output.stride(3);
    int64_t out_sW = output.stride(4);
    scalar_t *inp_ptr = input.data<scalar_t>();
    scalar_t *out_ptr = output.data<scalar_t>();
    scalar_t *grid_ptr = grid.data<scalar_t>();
    // loop over each output pixel
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t n = 0; n < N; ++n) {
      scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
      scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
      for (int64_t d = 0; d < out_D; ++d) {
        for (int64_t h = 0; h < out_H; ++h) {
          for (int64_t w = 0; w < out_W; ++w) {
            // get the corresponding input x, y, z co-ordinates from grid
            scalar_t *grid_ptr_NDHW = grid_ptr_N + d * grid_sD + h * grid_sH + w * grid_sW;
            scalar_t ix = *grid_ptr_NDHW;
            scalar_t iy = grid_ptr_NDHW[grid_sCoor];
            scalar_t iz = grid_ptr_NDHW[2 * grid_sCoor];

            // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
            ix = ((ix + 1) / 2) * (inp_W - 1);
            iy = ((iy + 1) / 2) * (inp_H - 1);
            iz = ((iz + 1) / 2) * (inp_D - 1);

            // get corner pixel values from (x, y, z)
            // for 4d, we used north-east-south-west
            // for 5d, we add top-bottom
            int64_t ix_tnw = static_cast<int64_t>(std::floor(ix));
            int64_t iy_tnw = static_cast<int64_t>(std::floor(iy));
            int64_t iz_tnw = static_cast<int64_t>(std::floor(iz));

            int64_t ix_tne = ix_tnw + 1;
            int64_t iy_tne = iy_tnw;
            int64_t iz_tne = iz_tnw;

            int64_t ix_tsw = ix_tnw;
            int64_t iy_tsw = iy_tnw + 1;
            int64_t iz_tsw = iz_tnw;

            int64_t ix_tse = ix_tnw + 1;
            int64_t iy_tse = iy_tnw + 1;
            int64_t iz_tse = iz_tnw;

            int64_t ix_bnw = ix_tnw;
            int64_t iy_bnw = iy_tnw;
            int64_t iz_bnw = iz_tnw + 1;

            int64_t ix_bne = ix_tnw + 1;
            int64_t iy_bne = iy_tnw;
            int64_t iz_bne = iz_tnw + 1;

            int64_t ix_bsw = ix_tnw;
            int64_t iy_bsw = iy_tnw + 1;
            int64_t iz_bsw = iz_tnw + 1;

            int64_t ix_bse = ix_tnw + 1;
            int64_t iy_bse = iy_tnw + 1;
            int64_t iz_bse = iz_tnw + 1;

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

            // calculate bilinear weighted pixel value and set output pixel
            scalar_t *out_ptr_NCDHW = out_ptr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
            scalar_t *inp_ptr_NC = inp_ptr_N;
            for (int c = 0; c < C; ++c, out_ptr_NCDHW += out_sC, inp_ptr_NC += inp_sC) {
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
      }
    }
    return output;
  }

  template<typename scalar_t>
  std::tuple<Tensor, Tensor>
  grid_sampler2d_backward_cpu_impl(const Tensor& grad_output,
                                   const Tensor& input, const Tensor& grid,
                                   GridSamplerInterpolation interpolation_mode,
                                   GridSamplerPadding padding_mode) {
    auto grad_input = at::zeros_like(input);
    auto grad_grid = at::empty_like(grid);
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_H = input.size(2);
    int64_t inp_W = input.size(3);
    int64_t out_H = grid.size(1);
    int64_t out_W = grid.size(2);
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sH = input.stride(2);
    int64_t inp_sW = input.stride(3);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sH = grid.stride(1);
    int64_t grid_sW = grid.stride(2);
    int64_t grid_sCoor = grid.stride(3);
    int64_t gOut_sN = grad_output.stride(0);
    int64_t gOut_sC = grad_output.stride(1);
    int64_t gOut_sH = grad_output.stride(2);
    int64_t gOut_sW = grad_output.stride(3);
    int64_t gInp_sN = grad_input.stride(0);
    int64_t gInp_sC = grad_input.stride(1);
    int64_t gInp_sH = grad_input.stride(2);
    int64_t gInp_sW = grad_input.stride(3);
    int64_t gGrid_sN = grad_grid.stride(0);
    int64_t gGrid_sW = grad_grid.stride(2);
    scalar_t *inp_ptr = input.data<scalar_t>();
    scalar_t *grid_ptr = grid.data<scalar_t>();
    scalar_t *gOut_ptr = grad_output.data<scalar_t>();
    scalar_t *gInp_ptr = grad_input.data<scalar_t>();
    scalar_t *gGrid_ptr = grad_grid.data<scalar_t>();
    // loop over each output pixel
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t n = 0; n < N; ++n) {
      scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
      scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
      scalar_t *gGrid_ptr_NHW = gGrid_ptr + n * gGrid_sN;
      for (int64_t h = 0; h < out_H; ++h) {
        for (int64_t w = 0; w < out_W; ++w, gGrid_ptr_NHW += gGrid_sW /* grad_grid is contiguous */ ) {
          // get the corresponding input x, y co-ordinates from grid
          scalar_t ix = grid_ptr_N[h * grid_sH + w * grid_sW];
          scalar_t iy = grid_ptr_N[h * grid_sH + w * grid_sW + grid_sCoor];

          // normalize ix, iy from [-1, 1] to [0, inp_W-1] & [0, inp_H-1]
          ix = ((ix + 1) / 2) * (inp_W - 1);
          iy = ((iy + 1) / 2) * (inp_H - 1);

          // get NE, NW, SE, SW pixel values from (x, y)
          int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
          int64_t iy_nw = static_cast<int64_t>(std::floor(iy));
          int64_t ix_ne = ix_nw + 1;
          int64_t iy_ne = iy_nw;
          int64_t ix_sw = ix_nw;
          int64_t iy_sw = iy_nw + 1;
          int64_t ix_se = ix_nw + 1;
          int64_t iy_se = iy_nw + 1;

          // get surfaces to each neighbor:
          scalar_t nw = (ix_se - ix)    * (iy_se - iy);
          scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
          scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
          scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

          int64_t ix_nw_cl, iy_nw_cl, ix_ne_cl, iy_ne_cl, ix_sw_cl, iy_sw_cl, ix_se_cl, iy_se_cl;

          if (padding_mode == GridSamplerPadding::Border) {
            // get clipped NE, NW, SE, SW pixel values from (x, y)
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
          scalar_t *gOut_ptr_NCHW = gOut_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
          scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
          scalar_t *inp_ptr_NC = inp_ptr_N;
          // calculate bilinear weighted pixel value and set output pixel
          for (int c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
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
          gix = gix * (inp_W - 1) / 2;
          giy = giy * (inp_H - 1) / 2;

          // assuming grad_grid is contiguous
          gGrid_ptr_NHW[0] = gix;
          gGrid_ptr_NHW[1] = giy;
        }
      }
    }
    return std::make_tuple(grad_input, grad_grid);
  }

  template<typename scalar_t>
  std::tuple<Tensor, Tensor>
  grid_sampler3d_backward_cpu_impl(const Tensor& grad_output,
                                   const Tensor& input, const Tensor& grid,
                                   GridSamplerInterpolation interpolation_mode,
                                   GridSamplerPadding padding_mode) {
    auto grad_input = at::zeros_like(input);
    auto grad_grid = at::empty_like(grid);
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_D = input.size(2);
    int64_t inp_H = input.size(3);
    int64_t inp_W = input.size(4);
    int64_t out_D = grid.size(1);
    int64_t out_H = grid.size(2);
    int64_t out_W = grid.size(3);
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sD = input.stride(2);
    int64_t inp_sH = input.stride(3);
    int64_t inp_sW = input.stride(4);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sD = grid.stride(1);
    int64_t grid_sH = grid.stride(2);
    int64_t grid_sW = grid.stride(3);
    int64_t grid_sCoor = grid.stride(4);
    int64_t gOut_sN = grad_output.stride(0);
    int64_t gOut_sC = grad_output.stride(1);
    int64_t gOut_sD = grad_output.stride(2);
    int64_t gOut_sH = grad_output.stride(3);
    int64_t gOut_sW = grad_output.stride(4);
    int64_t gInp_sN = grad_input.stride(0);
    int64_t gInp_sC = grad_input.stride(1);
    int64_t gInp_sD = grad_input.stride(2);
    int64_t gInp_sH = grad_input.stride(3);
    int64_t gInp_sW = grad_input.stride(4);
    int64_t gGrid_sN = grad_grid.stride(0);
    int64_t gGrid_sW = grad_grid.stride(3);
    scalar_t *inp_ptr = input.data<scalar_t>();
    scalar_t *grid_ptr = grid.data<scalar_t>();
    scalar_t *gOut_ptr = grad_output.data<scalar_t>();
    scalar_t *gInp_ptr = grad_input.data<scalar_t>();
    scalar_t *gGrid_ptr = grad_grid.data<scalar_t>();
    // loop over each output pixel
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t n = 0; n < N; ++n) {
      scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
      scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
      scalar_t *gGrid_ptr_NDHW = gGrid_ptr + n * gGrid_sN;
      for (int64_t d = 0; d < out_D; ++d) {
        for (int64_t h = 0; h < out_H; ++h) {
          for (int64_t w = 0; w < out_W; ++w, gGrid_ptr_NDHW += gGrid_sW /* grad_grid is contiguous */ ) {
            // get the corresponding input x, y, z co-ordinates from grid
            scalar_t *grid_ptr_NDHW = grid_ptr_N + d * grid_sD + h * grid_sH + w * grid_sW;
            scalar_t ix = *grid_ptr_NDHW;
            scalar_t iy = grid_ptr_NDHW[grid_sCoor];
            scalar_t iz = grid_ptr_NDHW[2 * grid_sCoor];

            // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
            ix = ((ix + 1) / 2) * (inp_W - 1);
            iy = ((iy + 1) / 2) * (inp_H - 1);
            iz = ((iz + 1) / 2) * (inp_D - 1);

            // get corner pixel values from (x, y, z)
            // for 4d, we used north-east-south-west
            // for 5d, we add top-bottom
            int64_t ix_tnw = static_cast<int64_t>(std::floor(ix));
            int64_t iy_tnw = static_cast<int64_t>(std::floor(iy));
            int64_t iz_tnw = static_cast<int64_t>(std::floor(iz));

            int64_t ix_tne = ix_tnw + 1;
            int64_t iy_tne = iy_tnw;
            int64_t iz_tne = iz_tnw;

            int64_t ix_tsw = ix_tnw;
            int64_t iy_tsw = iy_tnw + 1;
            int64_t iz_tsw = iz_tnw;

            int64_t ix_tse = ix_tnw + 1;
            int64_t iy_tse = iy_tnw + 1;
            int64_t iz_tse = iz_tnw;

            int64_t ix_bnw = ix_tnw;
            int64_t iy_bnw = iy_tnw;
            int64_t iz_bnw = iz_tnw + 1;

            int64_t ix_bne = ix_tnw + 1;
            int64_t iy_bne = iy_tnw;
            int64_t iz_bne = iz_tnw + 1;

            int64_t ix_bsw = ix_tnw;
            int64_t iy_bsw = iy_tnw + 1;
            int64_t iz_bsw = iz_tnw + 1;

            int64_t ix_bse = ix_tnw + 1;
            int64_t iy_bse = iy_tnw + 1;
            int64_t iz_bse = iz_tnw + 1;

            // get surfaces to each neighbor:
            scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
            scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
            scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
            scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
            scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
            scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
            scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
            scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

            int64_t ix_tnw_cl, iy_tnw_cl, iz_tnw_cl, ix_tne_cl, iy_tne_cl, iz_tne_cl;
            int64_t ix_tsw_cl, iy_tsw_cl, iz_tsw_cl, ix_tse_cl, iy_tse_cl, iz_tse_cl;
            int64_t ix_bnw_cl, iy_bnw_cl, iz_bnw_cl, ix_bne_cl, iy_bne_cl, iz_bne_cl;
            int64_t ix_bsw_cl, iy_bsw_cl, iz_bsw_cl, ix_bse_cl, iy_bse_cl, iz_bse_cl;

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
            scalar_t *gOut_ptr_NCDHW = gOut_ptr + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
            scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
            scalar_t *inp_ptr_NC = inp_ptr_N;
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
            gGrid_ptr_NDHW[0] = gix;
            gGrid_ptr_NDHW[1] = giy;
            gGrid_ptr_NDHW[2] = giz;
          }
        }
      }
    }
    return std::make_tuple(grad_input, grad_grid);
  }
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_2d_cpu(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode) {
  return AT_DISPATCH_FLOATING_TYPES(input.type(), "grid_sampler2d_cpu", [&] {
    return grid_sampler2d_cpu_impl<scalar_t>(
      input, grid, static_cast<GridSamplerInterpolation>(interpolation_mode),
      static_cast<GridSamplerPadding>(padding_mode));
  });
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_3d_cpu(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode) {
  return AT_DISPATCH_FLOATING_TYPES(input.type(), "grid_sampler3d_cpu", [&] {
    return grid_sampler3d_cpu_impl<scalar_t>(
      input, grid, static_cast<GridSamplerInterpolation>(interpolation_mode),
      static_cast<GridSamplerPadding>(padding_mode));
  });
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode) {
  return AT_DISPATCH_FLOATING_TYPES(input.type(), "grid_sampler_2d_backward_cpu", [&] {
    return grid_sampler2d_backward_cpu_impl<scalar_t>(
      grad_output, input, grid,
      static_cast<GridSamplerInterpolation>(interpolation_mode),
      static_cast<GridSamplerPadding>(padding_mode));
  });
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode) {
  return AT_DISPATCH_FLOATING_TYPES(input.type(), "grid_sampler_3d_backward_cpu", [&] {
    return grid_sampler3d_backward_cpu_impl<scalar_t>(
      grad_output, input, grid,
      static_cast<GridSamplerInterpolation>(interpolation_mode),
      static_cast<GridSamplerPadding>(padding_mode));
  });
}

Tensor grid_sampler(const Tensor& input, const Tensor& grid, int64_t padding_mode) {
  AT_CHECK(
    (input.dim() == 4 || input.dim() == 5) && input.dim() == grid.dim(),
    "grid_sampler(): expected 4D or 5D input and grid with same number "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
  AT_CHECK(
    input.size(0) == grid.size(0),
    "grid_sampler(): expected grid and input to have same batch size, but got "
    "input with sizes ", input.sizes(), " and grid with sizes ", grid.sizes());
  AT_CHECK(
    grid.size(-1) == input.dim() - 2,
    "grid_sampler(): expected grid to have size ", input.dim() - 2, " in last "
    "dimension, but got grid with sizes ", grid.sizes());
  // cudnn does not support inputs larger than 1024
  if (at::native::cudnn_is_acceptable(input) &&
      static_cast<GridSamplerPadding>(padding_mode) == GridSamplerPadding::Zeros &&
      input.dim() == 4 &&
      input.size(1) <= 1024) {
    return cudnn_grid_sampler(input, grid);
  }
  if (input.dim() == 4) {
    return at::grid_sampler_2d(input, grid, 0, padding_mode);
  } else {
    return at::grid_sampler_3d(input, grid, 0, padding_mode);
  }
}

}}  // namespace at::native

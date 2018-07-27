#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/detail/CUDAHooksInterface.h"
#include "ATen/native/GridSampler.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace at { namespace native {

namespace {
  static inline int64_t clip_coordinates(int64_t in, int64_t clip_limit) {
    return std::min(clip_limit - 1, std::max(in, static_cast<int64_t>(0)));
  }

  static inline bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W) {
    return h >= 0 && h < H && w >= 0 && w < W;
  }

  static inline bool within_bounds_3d(int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W) {
    return h >= 0 && h < H && w >= 0 && w < W;
  }

  template<typename scalar_t>
  static inline void safe_add_2d(scalar_t *data, int64_t h, int64_t w,
                                 int64_t sH, int64_t sW, int64_t H, int64_t W,
                                 scalar_t delta) {
    if (h >= 0 && h < H && w >= 0 && w < W) {
      data[h * sH + w * sW] += delta;
    }
  }

  template<typename scalar_t>
  static inline scalar_t safe_add_3d(scalar_t *data, int64_t d, int64_t h, int64_t w,
                                     int64_t sD, int64_t sH, int64_t sW,
                                     int64_t D, int64_t H, int64_t W,
                                     scalar_t delta) {
    if (d >=0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
      data[d * sD + h * sH + w * sW] += delta;
    }
  }

  template<typename scalar_t>
  Tensor grid_sampler2d_cpu_impl(const Tensor& input, const Tensor& grid,
                                 int64_t interpolation_mode, int64_t padding_mode) {
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
    int64_t out_sN = output.stride(0);
    int64_t out_sC = output.stride(1);
    int64_t out_sH = output.stride(2);
    int64_t out_sW = output.stride(3);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sH = grid.stride(1);
    int64_t grid_sW = grid.stride(2);
    int64_t grid_sCoor = grid.stride(3);
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

          // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
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

          if (padding_mode == detail::GridSamplerPaddingBorder) {
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
            if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
              *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
            }
            if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
              *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
            }
            if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
              *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
            }
            if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
              *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
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
                                   int64_t interpolation_mode, int64_t padding_mode) {
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

          // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
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

          if (padding_mode == detail::GridSamplerPaddingBorder) {
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
            if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_nw_cl, ix_nw_cl, inp_H, inp_W)) {
              scalar_t nw_val = inp_ptr_NC[iy_nw_cl * inp_sH + ix_nw_cl * inp_sW];
              gix -= nw_val * (iy_se - iy) * gOut;
              giy -= nw_val * (ix_se - ix) * gOut;
            }
            if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_ne_cl, ix_ne_cl, inp_H, inp_W)) {
              scalar_t ne_val = inp_ptr_NC[iy_ne_cl * inp_sH + ix_ne_cl * inp_sW];
              gix += ne_val * (iy_sw - iy) * gOut;
              giy -= ne_val * (ix - ix_sw) * gOut;
            }
            if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_sw_cl, ix_sw_cl, inp_H, inp_W)) {
              scalar_t sw_val = inp_ptr_NC[iy_sw_cl * inp_sH + ix_sw_cl * inp_sW];
              gix -= sw_val * (iy - iy_ne) * gOut;
              giy += sw_val * (ix_ne - ix) * gOut;
            }
            if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_se_cl, ix_se_cl, inp_H, inp_W)) {
              scalar_t se_val = inp_ptr_NC[iy_se_cl * inp_sH + ix_se_cl * inp_sW];
              gix += se_val * (iy - iy_nw) * gOut;
              giy += se_val * (ix - ix_nw) * gOut;
            }
          }

          // un-normalize grad_grid values back to [-1, 1] constraints
          gix = gix * (inp_W - 1) / 2;
          giy = giy * (inp_H - 1) / 2;

          gGrid_ptr_NHW[0] = gix;
          gGrid_ptr_NHW[1] = giy;
        }
      }
    }
    return std::make_tuple(grad_input, grad_grid);
  }
}

Tensor grid_sampler_2d_cpu(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode) {
  return AT_DISPATCH_FLOATING_TYPES(input.type(), "grid_sampler2d_cpu", [&] {
    return grid_sampler2d_cpu_impl<scalar_t>(input, grid, interpolation_mode, padding_mode);
  });
}

std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode) {
  return AT_DISPATCH_FLOATING_TYPES(input.type(), "grid_sampler_2d_backward_cpu", [&] {
    return grid_sampler2d_backward_cpu_impl<scalar_t>(grad_output, input, grid, interpolation_mode, padding_mode);
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
      padding_mode == detail::GridSamplerPaddingZeros &&
      input.dim() == 4 &&
      input.size(1) <= 1024) {
    return cudnn_grid_sampler(input, grid);
  }
  if (input.dim() == 4) {
    return at::grid_sampler_2d(input, grid, detail::GridSamplerInterpolationBilinear, padding_mode);
  } else {
    return thnn_grid_sampler_bilinear3d(input, grid, padding_mode);
  }
}

}}  // namespace at::native

#include <ATen/native/mps/kernels/GridSampler.h>
#include <ATen/native/mps/kernels/GridSamplerBackward.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;


template <typename scalar_t>
static inline scalar_t grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners) {
  if (align_corners) {
    return ((coord + 1.0f) / 2.0f) * (size - 1);
  } else {
    return ((coord + 1.0f) * size - 1.0f) / 2.0f;
  }
}

template <typename scalar_t>
static inline scalar_t grid_sampler_unnormalize_set_grad(scalar_t coord, int size,
                                                         bool align_corners, thread scalar_t* grad_in) {
  if (align_corners) {
    *grad_in = static_cast<scalar_t>(size - 1) / 2.0f;
    return ((coord + 1.0f) / 2.0f) * (size - 1);
  } else {
    *grad_in = static_cast<scalar_t>(size) / 2.0f;
    return ((coord + 1.0f) * size - 1.0f) / 2.0f;
  }
}

template <typename scalar_t>
static inline scalar_t clip_coordinates(scalar_t in, int clip_limit) {
  return clamp(in, static_cast<scalar_t>(0), static_cast<scalar_t>(clip_limit - 1));
}

template <typename scalar_t>
static inline scalar_t clip_coordinates_set_grad(scalar_t in, int clip_limit, thread scalar_t* grad_in) {
  if (in <= static_cast<scalar_t>(0)) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  } else {
    scalar_t max_val = static_cast<scalar_t>(clip_limit - 1);
    if (in >= max_val) {
      *grad_in = static_cast<scalar_t>(0);
      return max_val;
    } else {
      *grad_in = static_cast<scalar_t>(1);
      return in;
    }
  }
}

template <typename scalar_t>
static inline scalar_t reflect_coordinates(scalar_t in, int twice_low, int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min_val = static_cast<scalar_t>(twice_low) / 2.0f;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2.0f;
  in = abs(in - min_val);
  scalar_t extra = fmod(in, span);
  int flips = static_cast<int>(floor(in / span));
  if (flips % 2 == 0) {
    return extra + min_val;
  } else {
    return span - extra + min_val;
  }
}

template <typename scalar_t>
static inline scalar_t reflect_coordinates_set_grad(scalar_t in, int twice_low, int twice_high,
                                                    thread scalar_t* grad_in) {
  if (twice_low == twice_high) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  }
  int grad_in_mult;
  scalar_t min_val = static_cast<scalar_t>(twice_low) / 2.0f;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2.0f;
  in = in - min_val;
  if (in < static_cast<scalar_t>(0)) {
    grad_in_mult = -1;
    in = -in;
  } else {
    grad_in_mult = 1;
  }
  scalar_t extra = fmod(in, span);
  int flips = static_cast<int>(floor(in / span));
  if (flips % 2 == 0) {
    *grad_in = static_cast<scalar_t>(grad_in_mult);
    return extra + min_val;
  } else {
    *grad_in = static_cast<scalar_t>(-grad_in_mult);
    return span - extra + min_val;
  }
}

template<typename scalar_t>
static inline scalar_t safe_downgrade_to_int_range(scalar_t x){
  if (x > (numeric_limits<int>::max() - 1) || x < numeric_limits<int>::min() || !isfinite(x)) {
    return static_cast<scalar_t>(-100.0);
  }
  return x;
}

template<typename scalar_t>
static inline scalar_t compute_coordinates(scalar_t coord, int size,
                                           GridSamplerPadding padding_mode,
                                           bool align_corners) {
  // Note: Border padding mode is not supported on MPS and will be rejected by C++ code
  // before this kernel is called. The Border code path is kept here for completeness.
  coord = grid_sampler_unnormalize(coord, size, align_corners); // Forward needs unnormalize
  if (padding_mode == GridSamplerPadding::Border) {
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    coord = clip_coordinates(coord, size);
  }
  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

template <typename scalar_t>
static inline scalar_t grid_sampler_compute_source_index_set_grad(
    scalar_t coord, int size, GridSamplerPadding padding_mode,
    bool align_corners, thread scalar_t* grad_in) {
  // Note: Border padding mode is not supported on MPS (C++ code rejects it)
  scalar_t grad_clip, grad_refl;
  coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_in);
  if (padding_mode == GridSamplerPadding::Border) {
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    if (align_corners) {
      coord = reflect_coordinates_set_grad(coord, 0, 2 * (size - 1), &grad_refl);
    } else {
      coord = reflect_coordinates_set_grad(coord, -1, 2 * size - 1, &grad_refl);
    }
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_refl * grad_clip;
  }
  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

static inline bool within_bounds_2d(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template<typename scalar_t, typename index_t>
static inline void safe_add_2d(
    device scalar_t* data, int h, int w,
    int sH, int sW, int H, int W, scalar_t delta,
    const index_t NC_offset, const index_t memory_span) {
  if (within_bounds_2d(h, w, H, W)) {
    index_t offset = NC_offset + h * sH + w * sW;
    // Using atomic_float which is standard in Metal
    atomic_fetch_add_explicit(
      (device atomic_float*) data + offset,
      delta,
      memory_order_relaxed
    );
  }
}

template<typename scalar_t>
static inline scalar_t get_value_bounded(
    device const scalar_t* data, scalar_t x, scalar_t y, int W, int H, int sW, int sH,
    GridSamplerPadding padding_mode, bool align_corners) {
  // NOTE: This `compute_coordinates` is slightly different from backward pass one
  // as it doesn't return a gradient. It's used for fetching values.
  // Note: Border padding mode is not supported on MPS (C++ code rejects it)
  x = grid_sampler_unnormalize(x, W, align_corners);
  y = grid_sampler_unnormalize(y, H, align_corners);

  if (padding_mode == GridSamplerPadding::Border) {
    x = clip_coordinates(x, W);
    y = clip_coordinates(y, H);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    if (align_corners) {
      x = reflect_coordinates(x, 0, 2 * (W - 1));
      y = reflect_coordinates(y, 0, 2 * (H - 1));
    } else {
      x = reflect_coordinates(x, -1, 2 * W - 1);
      y = reflect_coordinates(y, -1, 2 * H - 1);
    }
    x = clip_coordinates(x, W);
    y = clip_coordinates(y, H);
  }
  x = safe_downgrade_to_int_range(x);
  y = safe_downgrade_to_int_range(y);

  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);
  if (within_bounds_2d(iy, ix, H, W)) {
    return data[iy * sH + ix * sW];
  }
  return static_cast<scalar_t>(0);
}


template<typename scalar_t, typename index_t>
static inline void add_value_bounded(
    device scalar_t* data, scalar_t ix, scalar_t iy, int W, int H, int sW, int sH,
    scalar_t delta, GridSamplerPadding padding_mode, bool align_corners,
    const index_t NC_offset, const index_t memory_span) {
  // In the backward pass for bicubic, ix/iy are already unnormalized,
  // but we still need to handle padding before casting to int.
  ix = compute_coordinates(ix, W, padding_mode, align_corners);
  iy = compute_coordinates(iy, H, padding_mode, align_corners);

  int ix_int = static_cast<int>(ix);
  int iy_int = static_cast<int>(iy);
  safe_add_2d(data, iy_int, ix_int, sH, sW, H, W, delta, NC_offset, memory_span);
}



template <typename scalar_t>
kernel void grid_sampler_2d_backward(
  device scalar_t* grad_input [[buffer(0)]],
  device scalar_t* grad_grid [[buffer(1)]],
  const device scalar_t* grad_output [[buffer(2)]],
  const device scalar_t* input [[buffer(3)]],
  const device scalar_t* grid [[buffer(4)]],
  const device GridSamplerBackwardParams<5>& params [[buffer(5)]],
  uint tid [[thread_position_in_grid]])
{
  
  using index_t = int32_t;

  const bool align_corners = params.align_corners;
  const GridSamplerInterpolation interpolation_mode = params.interpolation_mode;
  const GridSamplerPadding padding_mode = params.padding_mode;
  const bool input_requires_grad = params.input_requires_grad;

  
  index_t C = params.input_sizes[1];
  index_t inp_H = params.input_sizes[2];
  index_t inp_W = params.input_sizes[3];
  index_t out_H = params.grid_sizes[1];
  index_t out_W = params.grid_sizes[2];
  index_t inp_sN = params.input_strides[0];
  index_t inp_sC = params.input_strides[1];
  index_t inp_sH = params.input_strides[2];
  index_t inp_sW = params.input_strides[3];
  index_t grid_sN = params.grid_strides[0];
  index_t grid_sH = params.grid_strides[1];
  index_t grid_sW = params.grid_strides[2];
  index_t grid_sCoor = params.grid_strides[3];
  index_t gOut_sN = params.grad_output_strides[0];
  index_t gOut_sC = params.grad_output_strides[1];
  index_t gOut_sH = params.grad_output_strides[2];
  index_t gOut_sW = params.grad_output_strides[3];

  // gInp_* (and NC_offset below) are not really needed if input_requires_grad is false.
  index_t gInp_sN = 0, gInp_sC = 0, gInp_sH = 0, gInp_sW = 0;
  if (input_requires_grad) {
    gInp_sN = params.grad_input_strides[0];
    gInp_sC = params.grad_input_strides[1];
    gInp_sH = params.grad_input_strides[2];
    gInp_sW = params.grad_input_strides[3];
  }
  index_t gGrid_sW = params.grad_grid_strides[2];
  const index_t grad_input_memory_span = params.grad_input_memory_span;
  
  index_t index = tid ; 
    const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t n = index / (out_H * out_W);
      const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t x = grid[grid_offset];
      scalar_t y = grid[grid_offset + grid_sCoor];

      // multipliers for gradients on ix and iy
      scalar_t gix_mult, giy_mult;
      scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);
      scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        index_t ix_nw = static_cast<index_t>(floor(ix));
        index_t iy_nw = static_cast<index_t>(floor(iy));
        index_t ix_ne = ix_nw + 1;
        index_t iy_ne = iy_nw;
        index_t ix_sw = ix_nw;
        index_t iy_sw = iy_nw + 1;
        index_t ix_se = ix_nw + 1;
        index_t iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix)    * (iy_se - iy);
        scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
        scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
        scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
        device const scalar_t *gOut_ptr_NCHW = grad_output + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        index_t NC_offset = n * gInp_sN;
        device const scalar_t *inp_ptr_NC = input + n * inp_sN;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
          const scalar_t gOut = *gOut_ptr_NCHW;

          if (input_requires_grad) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_2d(grad_input, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut, NC_offset, grad_input_memory_span);
          }

          // calculate grad_grid
          if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
            gix -= nw_val * (iy_se - iy) * gOut;
            giy -= nw_val * (ix_se - ix) * gOut;
          }
          if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
            gix += ne_val * (iy_sw - iy) * gOut;
            giy -= ne_val * (ix - ix_sw) * gOut;
          }
          if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
            gix -= sw_val * (iy - iy_ne) * gOut;
            giy += sw_val * (ix_ne - ix) * gOut;
          }
          if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
            gix += se_val * (iy - iy_nw) * gOut;
            giy += se_val * (ix - ix_nw) * gOut;
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        device scalar_t *gGrid_ptr_NHW = grad_grid + index * gGrid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix;
        gGrid_ptr_NHW[1] = giy_mult * giy;
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        if (input_requires_grad) {
          index_t ix_nearest = (index_t)rint(ix);
          index_t iy_nearest = (index_t)rint(iy);


          // assign nearest neighbour pixel value to output pixel
          device const scalar_t *gOut_ptr_NCHW = grad_output + n * gOut_sN + h * gOut_sH + w * gOut_sW;
          index_t NC_offset = n * gInp_sN;
          for (index_t c = 0; c < C; ++c, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_2d(grad_input, iy_nearest, ix_nearest, gInp_sH, gInp_sW, inp_H, inp_W, *gOut_ptr_NCHW, NC_offset, grad_input_memory_span);
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        device scalar_t *gGrid_ptr_NHW = grad_grid + index * gGrid_sW;
        gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
        gGrid_ptr_NHW[1] = static_cast<scalar_t>(0);
      }
      // Note: Bicubic interpolation is not supported on MPS
      // The C++ code will reject it before reaching this kernel
    
}

#define REGISTER_GRID_SAMPLER_OP(DTYPE)                     \
  template [[host_name("grid_sampler_2d_backward_" #DTYPE)]]            \
  kernel void grid_sampler_2d_backward<DTYPE>( \
      device DTYPE* grad_input [[buffer(0)]],\
      device DTYPE* grad_grid [[buffer(1)]],\
      const device DTYPE* grad_output [[buffer(2)]],\
      const device DTYPE* input [[buffer(3)]],\
      const device DTYPE* grid [[buffer(4)]],\
      const device GridSamplerBackwardParams<5>& params [[buffer(5)]],\
      uint tid [[thread_position_in_grid]]);

REGISTER_GRID_SAMPLER_OP(float);
REGISTER_GRID_SAMPLER_OP(half);
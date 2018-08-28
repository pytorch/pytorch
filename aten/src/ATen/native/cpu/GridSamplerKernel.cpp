#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/C++17.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/GridSampler.h>
#include <ATen/native/cpu/GridSamplerKernel.h>
#include <ATen/cpu/vml.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <iostream>
#include <bitset>

namespace at { namespace native { namespace {

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;
using namespace at::vec256;

// clip_coordinates_set_grad works similarly to clip_coordinates except that
// it also returns the `d output / d input` via pointer argument `grad_in`.
// This is useful in the backward pass of grid_sampler.
template<typename scalar_t>
struct ApplyPaddingBase {
  const Vec256<scalar_t> half_max_val;
  const Vec256<scalar_t> zeros = Vec256<scalar_t>(0);

  ApplyPaddingBase(int64_t size)
    : half_max_val(static_cast<scalar_t>(size - 1) / 2) {}

  inline Vec256<scalar_t> unnormalize(const Vec256<scalar_t> &in) const {
    return (in + Vec256<scalar_t>(1)) * half_max_val;
  }
};

template<typename scalar_t, GridSamplerPadding padding>
struct ApplyPadding;

template<typename scalar_t>
struct ApplyPadding<scalar_t, GridSamplerPadding::Zeros>
  : ApplyPaddingBase<scalar_t> {

  using ApplyPaddingBase<scalar_t>::ApplyPaddingBase;

  inline Vec256<scalar_t> apply(const Vec256<scalar_t> &in) const {
    return this->unnormalize(in);
  }

  inline std::pair<Vec256<scalar_t>, Vec256<scalar_t>> apply_get_grad(const Vec256<scalar_t> &in) const {
    return std::make_pair(this->unnormalize(in), this->half_max_val);
  }
};

template<typename scalar_t>
struct ApplyPadding<scalar_t, GridSamplerPadding::Border>
  : ApplyPaddingBase<scalar_t> {

  const Vec256<scalar_t> max_val;

  ApplyPadding(int64_t size)
    : ApplyPaddingBase<scalar_t>(size)
    , max_val(static_cast<scalar_t>(size - 1)) {}

  inline Vec256<scalar_t> apply(const Vec256<scalar_t> &in) const {
    return min(this->max_val, max(this->unnormalize(in), this->zeros));
  }
  inline std::pair<Vec256<scalar_t>, Vec256<scalar_t>> apply_get_grad(const Vec256<scalar_t> &in) const {
    AT_ERROR("FIXME: NYI");
    return std::make_pair(apply(in), this->zeros);
  }
};

template<typename scalar_t>
struct ApplyPadding<scalar_t, GridSamplerPadding::Reflection>
  : ApplyPaddingBase<scalar_t> {

  bool unit_size;  // whether size == 1, just return 0 in this case
  const Vec256<scalar_t> double_max_val;

  ApplyPadding(int64_t size)
    : ApplyPaddingBase<scalar_t>(size)
    , unit_size(size == 1)
    , double_max_val(static_cast<scalar_t>((size - 1) * 2)) {}

  inline Vec256<scalar_t> apply(const Vec256<scalar_t> &in) const {
    if (unit_size) {
      return this->zeros;
    }
    auto abs_in = this->unnormalize(in).abs();
    auto fdouble_flips = abs_in / double_max_val;
    auto double_flips = fdouble_flips.trunc();
    auto extra = abs_in - double_flips * double_max_val;
    return min(extra, double_max_val - extra);
  }

  inline std::pair<Vec256<scalar_t>, Vec256<scalar_t>> apply_get_grad(const Vec256<scalar_t> &in) const {
    AT_ERROR("FIXME: NYI");
    return std::make_pair(this->unnormalize(in), this->zeros);
  }
};


template<typename scalar_t, int dim, GridSamplerInterpolation interp>
struct ApplyInterpolation;

template<typename scalar_t>
struct ApplyInterpolation<scalar_t, 2, GridSamplerInterpolation::Bilinear> {
  using Vec = Vec256<scalar_t>;
  using iVec = Vec256<int_same_size_t<scalar_t>>;

  const iVec i_H;
  const iVec i_W;
  const iVec i_inp_sH;
  const iVec i_inp_sW;
  const iVec i_neg1s = iVec(-1);
  const iVec i_ones = iVec(1);
  const Vec ones = Vec(1);
  const Vec zeros = Vec(0);
  int64_t C;
  int64_t inp_sC;

  ApplyInterpolation(const TensorAccessor<scalar_t, 4>& input)
    : i_H(iVec(input.size(2))), i_W(iVec(input.size(3)))
    , i_inp_sH(input.stride(2)), i_inp_sW(input.stride(3))
    , C(input.size(1)), inp_sC(input.stride(1)) {}

  inline void apply(scalar_t *out_ptr, const scalar_t *inp_slice_ptr,
                    int64_t out_sC, const Vec& x, const Vec& y,
                    bool must_in_bound, int64_t len) const {
    // get NE, NW, SE, SW pixel values from (x, y)
    // assuming we get exact integer representation and just use scalar_t
    // if we don't, the weights will be garbage anyways.
    auto x_w = x.floor();
    auto y_n = y.floor();
    auto x_e = x_w + ones;
    auto y_s = y_n + ones;

    // get surfaces to each neighbor:
    auto nw = (x_e - x)   * (y_s - y);
    auto ne = (x   - x_w) * (y_s - y);
    auto sw = (x_e - x)   * (y   - y_n);
    auto se = (x   - x_w) * (y   - y_n);

    auto i_x_w = convert_to_int_of_same_size(x_w);
    auto i_y_n = convert_to_int_of_same_size(y_n);
    // int addition is faster than float -> int conversion
    auto i_x_e = i_x_w + i_ones;
    auto i_y_s = i_y_n + i_ones;

    // Use int comparison because it is much faster than float comp with AVX2
    // (latency 1 cyc vs. 4 cyc on skylake)
    // Avoid using the le and ge because those are not implemented in AVX2 and
    // are actually simulated using multiple instructions.
    auto w_mask = must_in_bound ? i_neg1s  // true = all ones
                                : i_x_w.gt(i_neg1s) & i_x_w.lt(i_W);
    auto n_mask = must_in_bound ? i_neg1s  // true = all ones
                                : i_y_n.gt(i_neg1s) & i_y_n.lt(i_H);
    auto e_mask = must_in_bound ? i_x_e.lt(i_W)
                                : i_x_e.gt(i_neg1s) & i_x_e.lt(i_W);
    auto s_mask = must_in_bound ? i_y_s.lt(i_H)
                                : i_y_s.gt(i_neg1s) & i_y_s.lt(i_H);
    auto nw_mask = cast<scalar_t>(must_in_bound ? i_neg1s : (w_mask & n_mask));
    auto sw_mask = cast<scalar_t>(w_mask & s_mask);
    auto ne_mask = cast<scalar_t>(e_mask & n_mask);
    auto se_mask = cast<scalar_t>(e_mask & s_mask);

    auto i_nw_offset = i_y_n * i_inp_sH + i_x_w * i_inp_sW;
    auto i_ne_offset = i_nw_offset + i_inp_sW;
    auto i_sw_offset = i_nw_offset + i_inp_sH;
    auto i_se_offset = i_sw_offset + i_inp_sW;

    #pragma unroll
    for (int c = 0; c < C; ++c, out_ptr += out_sC, inp_slice_ptr += inp_sC) {
      // mask_gather zeros out the mask, so we need to make copies
      Vec nw_mask_copy = nw_mask;
      Vec ne_mask_copy = ne_mask;
      Vec sw_mask_copy = sw_mask;
      Vec se_mask_copy = se_mask;
      auto nw_inp_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_nw_offset, nw_mask_copy);
      auto ne_inp_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_ne_offset, ne_mask_copy);
      auto sw_inp_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_sw_offset, sw_mask_copy);
      auto se_inp_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_se_offset, se_mask_copy);
      auto interpolated = (nw_inp_val * nw) + (ne_inp_val * ne) +
                          (sw_inp_val * sw) + (se_inp_val * se);
      interpolated.store(static_cast<void*>(out_ptr), len);
    }
  }
};

template<typename scalar_t>
struct ApplyInterpolation<scalar_t, 2, GridSamplerInterpolation::Nearest> {
  using Vec = Vec256<scalar_t>;
  using iVec = Vec256<int_same_size_t<scalar_t>>;

  const iVec i_H;
  const iVec i_W;
  const iVec i_inp_sH;
  const iVec i_inp_sW;
  const iVec i_neg1s = iVec(-1);
  const iVec i_ones = iVec(-1);
  const Vec ones = Vec(1);
  const Vec zeros = Vec(0);
  int64_t C;
  int64_t inp_sC;

  ApplyInterpolation(const TensorAccessor<scalar_t, 4>& input)
    : i_H(iVec(input.size(2))), i_W(iVec(input.size(3)))
    , i_inp_sH(input.stride(2)), i_inp_sW(input.stride(3))
    , C(input.size(1)), inp_sC(input.stride(1)) {}

  inline void apply(scalar_t *out_ptr, const scalar_t *inp_slice_ptr,
                    int64_t out_sC, const Vec& x, const Vec& y,
                    bool must_in_bound, int64_t len) const {
    auto x_nearest = x.round();
    auto y_nearest = y.round();

    auto i_x_nearest = convert_to_int_of_same_size(x_nearest);
    auto i_y_nearest = convert_to_int_of_same_size(y_nearest);

    auto i_mask = must_in_bound ? i_neg1s
                                : (i_x_nearest.gt(i_neg1s) & i_x_nearest.lt(i_W) &
                                   i_y_nearest.gt(i_neg1s) & i_y_nearest.lt(i_H));
    auto mask = cast<scalar_t>(i_mask);

    auto i_offset = i_y_nearest * i_inp_sH + i_x_nearest * i_inp_sW;

    #pragma unroll
    for (int c = 0; c < C; ++c, out_ptr += out_sC, inp_slice_ptr += inp_sC) {
      // mask_gather zeros out the mask, so we need to make a copy
      auto mask_copy = mask;
      auto inp_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_offset, mask_copy);
      inp_val.store(static_cast<void*>(out_ptr), len);
    }
  }
};

template<typename scalar_t,
         GridSamplerInterpolation interp_mode,
         GridSamplerPadding padding_mode>
struct ApplyForward2d {
  ApplyPadding<scalar_t, padding_mode> &padH;
  ApplyPadding<scalar_t, padding_mode> &padW;
  ApplyInterpolation<scalar_t, 2, interp_mode> &interp;

  const int64_t out_sC;
  const scalar_t *inp_slice_ptr;
  scalar_t *out_slice_ptr;

  ApplyForward2d(ApplyPadding<scalar_t, padding_mode> &padH,
                 ApplyPadding<scalar_t, padding_mode> &padW,
                 ApplyInterpolation<scalar_t, 2, interp_mode> &interp,
                 const TensorAccessor<scalar_t, 3> inp_slice,
                 TensorAccessor<scalar_t, 3> out_slice)
    : padH(padH)
    , padW(padW)
    , interp(interp)
    , out_sC(out_slice.stride(0))
    , inp_slice_ptr(inp_slice.data())
    , out_slice_ptr(out_slice.data()) {}

  inline void operator()(const Vec256<scalar_t>& grid_ix,
                         const Vec256<scalar_t>& grid_iy,
                         int64_t spatial_offset, int64_t len) {
    auto fix = padW.apply(grid_ix);
    auto fiy = padH.apply(grid_iy);
    auto always_in_bound = padding_mode != GridSamplerPadding::Zeros;
    interp.apply(out_slice_ptr + spatial_offset,
                inp_slice_ptr, out_sC, fix, fiy, always_in_bound, len);
  }
};

template<typename scalar_t, typename GetApply2d>
static inline void grid_sample_2d_grid_iterator (
    const TensorAccessor<scalar_t, 4>& grid, const GetApply2d &get_apply_fn) {
  int64_t N = grid.size(0);
  int64_t out_H = grid.size(1);
  int64_t out_W = grid.size(2);
  int64_t grid_sN = grid.stride(0);
  int64_t grid_sH = grid.stride(1);
  int64_t grid_sW = grid.stride(2);
  int64_t grid_sCoor = grid.stride(3);
  auto grid_ptr = grid.data();

  using Vec = Vec256<scalar_t>;
  using iVec = Vec256<int_same_size_t<scalar_t>>;
  const int64_t step = Vec::size;

  // loop over each output pixel
  #ifdef _OPENMP
  #pragma parallel for if (N > 1)
  #endif
  for (int64_t n = 0; n < N; ++n) {
    const scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;

    auto apply_fn = get_apply_fn(n);

    // For the three tensors we work with, input, grid, output, we know output
    // is contiguous, and we need to do random read for input anyways. So we
    // base our iterating strategy on the geometry of grid. We consider the
    // following three cases (after slicing out the batch dimension).
    // See detailed discussions under each if-case.

    if (at::geometry_is_contiguous({out_H, out_W, 2}, {grid_sH, grid_sW, grid_sCoor})) {
      // Case 1:
      // Grid is contiguous.
      // Strategy: Sequentially load two vectors at the same time, and get,
      //           e.g.,  {x0, y0, x1, y1}, {x2, y2, x3, y3}. Then we use
      //           at::vec256::deinterleave2 to get x and y vectors.
      auto total_size = out_H * out_W;
      for (int64_t offset = 0; offset < total_size; offset += step) {
        auto grid_offset = offset * 2;
        auto len = std::min(step, total_size - offset);
        auto vec1 = Vec::loadu(grid_ptr_N + grid_offset,
                               std::min(step, len * 2));
        auto vec2 = Vec::loadu(grid_ptr_N + grid_offset + step,
                               std::max(static_cast<int64_t>(0), len * 2 - step));
        auto vec_xy_pair = deinterleave2(vec1, vec2);
        apply_fn(std::get<0>(vec_xy_pair), std::get<1>(vec_xy_pair),
                 offset, len);
      }
    } else if (grid_sW == 1 || out_W == 1) {
      // Case 2:
      // The W dimension is contiguous.
      // This can be common, e.g., grid is from a conv net output of shape
      // [N, 2, H, W].
      // Strategy: Divide into two contiguous slices each of shape [H, W], and
      //           each containing x and y vectors. So we sequentially load a
      //           vector from each of them to get x and y vector

      // Function to apply along a contiguous W dimension (or flattened H x W).
      auto line_fn = [&](const scalar_t *grid_ptr_x, const scalar_t *grid_ptr_y,
                         int64_t out_base_offset, int64_t total_size) {
        for (int64_t i = 0; i < total_size; i += step) {
          auto len = std::min(step, total_size - i);
          apply_fn(Vec::loadu(grid_ptr_x + i, len),
                   Vec::loadu(grid_ptr_y + i, len),
                   out_base_offset + i, len);
        }
      };

      if (at::geometry_is_contiguous({out_H, out_W}, {grid_sH, grid_sW})) {
        // If [H, W] is contiguous, apply line_fn once.
        line_fn(grid_ptr_N, grid_ptr_N + grid_sCoor, 0, out_H * out_W);
      } else {
        // If only [W] is contiguous, apply line_fn once for each h slice.
        auto grid_ptr_NH = grid_ptr_N;
        for (int64_t h = 0; h < out_H; h++) {
          line_fn(grid_ptr_NH, grid_ptr_NH + grid_sCoor, h * grid_sH, out_W);
          grid_ptr_NH += grid_sH;
        }
      }
    } else {
      // Case 3:
      // General case.
      // Strategy: Do a for-loop over H, for each W slice, use
      //           at::vec256::gather to load the x and y vectors.
      auto i_zeros = iVec(0);
      auto spatial_offset = 0;
      auto i_offsets_delta = iVec(grid_sW * step);

      #pragma unroll
      for (int64_t h = 0; h < out_H; h++) {
        auto grid_ptr_x = grid_ptr_N + h * grid_sH;
        auto grid_ptr_y = grid_ptr_x + grid_sCoor;
        auto i_offsets = iVec::arange(0, grid_sW);
        #pragma unroll
        for (int64_t w = 0; w < out_W; w += step) {
          auto len = std::min(step, out_W - w);
          if (len < step) {
            // prevents illegal memory access, sets the exceeding offsets to zero
            i_offsets = iVec::set(i_zeros, i_offsets, len);
          }

          apply_fn(gather<sizeof(scalar_t)>(grid_ptr_x, i_offsets),
                   gather<sizeof(scalar_t)>(grid_ptr_y, i_offsets),
                   spatial_offset, len);

          i_offsets = i_offsets + i_offsets_delta;
          spatial_offset += len;
        }
      }
    }
  }
}

template<typename scalar_t,
         GridSamplerInterpolation interp_mode,
         GridSamplerPadding padding_mode>
static inline
void grid_sample_2d_vec_kernel(Tensor& output, const Tensor& input, const Tensor& grid) {

  auto inp_acc = input.accessor<scalar_t, 4>();
  auto grid_acc = grid.accessor<scalar_t, 4>();
  auto out_acc = output.accessor<scalar_t, 4>();

  ApplyPadding<scalar_t, padding_mode> padH(inp_acc.size(2));
  ApplyPadding<scalar_t, padding_mode> padW(inp_acc.size(3));
  ApplyInterpolation<scalar_t, 2, interp_mode> interp(inp_acc);

  grid_sample_2d_grid_iterator(
    grid_acc,
    [&](int64_t n) {
      return ApplyForward2d<scalar_t, interp_mode, padding_mode>(
          padH, padW, interp, inp_acc[n], out_acc[n]);
    });
}

Tensor grid_sampler_2d_cpu_kernel_impl(const Tensor& input, const Tensor& grid,
               int64_t interpolation_mode, int64_t padding_mode) {
  auto output = at::empty({input.size(0), input.size(1), grid.size(1), grid.size(2)}, input.options());

#define HANDLE_CASE(interp, padding)                                            \
  case padding: {                                                               \
    grid_sample_2d_vec_kernel<scalar_t, interp, padding>(output, input, grid);  \
    return;                                                                     \
  }

#define HANDLE_INTERP(interp)                                          \
  case interp: {                                                       \
    switch (static_cast<GridSamplerPadding>(padding_mode)) {           \
      HANDLE_CASE(interp, GridSamplerPadding::Zeros);                  \
      HANDLE_CASE(interp, GridSamplerPadding::Border);                 \
      HANDLE_CASE(interp, GridSamplerPadding::Reflection);             \
    }                                                                  \
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "grid_sampler_2d_cpu_kernel_impl", [&] {
    switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
      HANDLE_INTERP(GridSamplerInterpolation::Bilinear);
      HANDLE_INTERP(GridSamplerInterpolation::Nearest);
    }
  });
#undef HANDLE_CASE
#undef HANDLE_INTERP

  return output;
}

}

REGISTER_DISPATCH(grid_sampler_2d_cpu_kernel, &grid_sampler_2d_cpu_kernel_impl);


}}  // namespace at::native

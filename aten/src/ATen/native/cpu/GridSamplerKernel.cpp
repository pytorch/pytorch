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

namespace at { namespace native { namespace {

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;
using namespace at::vec256;

// clip_coordinates_set_grad works similarly to clip_coordinates except that
// it also returns the `d output / d input` via pointer argument `grad_in`.
// This is useful in the backward pass of grid_sampler.
template<typename scalar_t>
struct ComputeLocationBase {
  using Vec = Vec256<scalar_t>;

  const Vec half_max_val;
  const Vec zeros = Vec(0);
  const Vec ones = Vec(1);

  ComputeLocationBase(int64_t size)
    : half_max_val(static_cast<scalar_t>(size - 1) / 2) {}

  inline Vec unnormalize(const Vec &in) const {
    return (in + ones) * half_max_val;
  }
};

template<typename scalar_t, GridSamplerPadding padding>
struct ComputeLocation;

template<typename scalar_t>
struct ComputeLocation<scalar_t, GridSamplerPadding::Zeros>
  : ComputeLocationBase<scalar_t> {
  using Vec = Vec256<scalar_t>;
  using ComputeLocationBase<scalar_t>::unnormalize;
  using ComputeLocationBase<scalar_t>::half_max_val;

  using ComputeLocationBase<scalar_t>::ComputeLocationBase;

  inline Vec apply(const Vec &in) const {
    return unnormalize(in);
  }

  inline std::pair<Vec, Vec> apply_get_grad(const Vec &in) const {
    return std::make_pair(unnormalize(in), half_max_val);
  }
};

template<typename scalar_t>
struct ComputeLocation<scalar_t, GridSamplerPadding::Border>
  : ComputeLocationBase<scalar_t> {
  using Vec = Vec256<scalar_t>;
  using ComputeLocationBase<scalar_t>::zeros;
  using ComputeLocationBase<scalar_t>::unnormalize;
  using ComputeLocationBase<scalar_t>::half_max_val;

  const Vec max_val;

  ComputeLocation(int64_t size)
    : ComputeLocationBase<scalar_t>(size)
    , max_val(static_cast<scalar_t>(size - 1)) {}

  inline Vec apply(const Vec &in) const {
    return min(max_val, max(unnormalize(in), zeros));
  }
  inline std::pair<Vec, Vec> apply_get_grad(const Vec &in) const {
    auto indices = unnormalize(in);
    auto in_bound_hi = indices <= max_val;
    auto in_bound_lo = indices >= zeros;
    auto res = Vec::blendv(zeros,
                           Vec::blendv(max_val, indices, in_bound_hi),
                           in_bound_lo);
    return std::make_pair(res, (in_bound_hi & in_bound_lo) & half_max_val);
  }
};

template<typename scalar_t>
struct ComputeLocation<scalar_t, GridSamplerPadding::Reflection>
  : ComputeLocationBase<scalar_t> {
  using Vec = Vec256<scalar_t>;
  using ComputeLocationBase<scalar_t>::zeros;
  using ComputeLocationBase<scalar_t>::unnormalize;
  using ComputeLocationBase<scalar_t>::half_max_val;

  bool unit_size;  // whether size == 1, just return 0 in this case
  const Vec double_max_val;
  const Vec neg_half_max_val;

  ComputeLocation(int64_t size)
    : ComputeLocationBase<scalar_t>(size)
    , unit_size(size == 1)
    , double_max_val(static_cast<scalar_t>((size - 1) * 2))
    , neg_half_max_val(-0.5 * static_cast<scalar_t>(size - 1)) {}

  inline Vec apply(const Vec &in) const {
    if (unit_size) {
      return zeros;
    }
    auto abs_in = unnormalize(in).abs();
    auto fdouble_flips = abs_in / double_max_val;
    auto double_flips = fdouble_flips.trunc();
    auto extra = abs_in - double_flips * double_max_val;
    // Now we need to test if extra > max_val to find out if another flip is
    // needed. The following comparison does that and returns the correct
    // flipped value.
    return min(extra, double_max_val - extra);
  }

  inline std::pair<Vec, Vec> apply_get_grad(const Vec &in) const {
    if (unit_size) {
      return std::make_pair(zeros, zeros);
    }
    auto unnorm_in = unnormalize(in);
    auto neg_in = unnorm_in < zeros;
    auto abs_in = unnorm_in.abs();
    auto fdouble_flips = abs_in / double_max_val;
    auto double_flips = fdouble_flips.trunc();

    auto extra = abs_in - double_flips * double_max_val;
    auto reflected_extra = double_max_val - extra;
    auto one_more_flip = extra > reflected_extra;

    return std::make_pair(
      Vec::blendv(extra, reflected_extra, one_more_flip),
      Vec::blendv(half_max_val, neg_half_max_val, one_more_flip ^ neg_in)
    );
  }
};

template<typename scalar_t>
static inline void
mask_scatter_add(const scalar_t *src, scalar_t* base_addr,
                 const int_same_size_t<scalar_t> *offsets,
                 const int_same_size_t<scalar_t> *mask, int64_t len) {
  #pragma unroll
  for (int64_t i = 0; i < len; i++) {
    if (mask[i] & 0x01) {
      base_addr[offsets[i]] += src[i];
    }
  }
}

template<typename scalar_t, int dim,
         GridSamplerPadding padding,
         GridSamplerInterpolation interp>
struct ApplyGridSample;

template<typename scalar_t, GridSamplerPadding padding>
struct ApplyGridSample<scalar_t, 2, padding, GridSamplerInterpolation::Bilinear> {
  using Vec = Vec256<scalar_t>;
  using integer_t = int_same_size_t<scalar_t>;
  using iVec = Vec256<integer_t>;

  const iVec i_inp_H;
  const iVec i_inp_W;
  const iVec i_inp_sH;
  const iVec i_inp_sW;
  const iVec i_neg1s = iVec(-1);
  const iVec i_ones = iVec(1);
  const Vec ones = Vec(1);
  const Vec zeros = Vec(0);
  int64_t C;
  int64_t inp_sC;
  const ComputeLocation<scalar_t, padding> compute_H;
  const ComputeLocation<scalar_t, padding> compute_W;
  const bool must_in_bound = padding != GridSamplerPadding::Zeros;

  ApplyGridSample(const TensorAccessor<scalar_t, 4>& input)
    : i_inp_H(input.size(2))
    , i_inp_W(input.size(3))
    , i_inp_sH(input.stride(2))
    , i_inp_sW(input.stride(3))
    , C(input.size(1))
    , inp_sC(input.stride(1))
    , compute_H(input.size(2))
    , compute_W(input.size(3)) {}

  inline std::tuple<
    Vec, Vec, Vec, Vec,       // interpolation weights wrt 4 sides
    Vec, Vec, Vec, Vec,       // interpolation weights wrt 4 corners
    Vec, Vec, Vec, Vec,       // in_bound masks
    iVec, iVec                // y_n and x_w
  >
  compute_interp_params(const Vec& x, const Vec& y) const {
    // get NE, NW, SE, SW pixel values from (x, y)
    // assuming we get exact integer representation and just use scalar_t
    // if we don't, the weights will be garbage anyways.
    auto x_w = x.floor();
    auto y_n = y.floor();

    // get surfaces to each neighbor:
    auto w = x - x_w;
    auto e = ones - w;
    auto n = y - y_n;
    auto s = ones - n;

    auto nw = s * e;
    auto ne = s * w;
    auto sw = n * e;
    auto se = n * w;

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
                                : (i_x_w > i_neg1s) & (i_x_w < i_inp_W);
    auto n_mask = must_in_bound ? i_neg1s  // true = all ones
                                : (i_y_n > i_neg1s) & (i_y_n < i_inp_H);
    auto e_mask = must_in_bound ? (i_x_e < i_inp_W)
                                : (i_x_e > i_neg1s) & (i_x_e < i_inp_W);
    auto s_mask = must_in_bound ? (i_y_s < i_inp_H)
                                : (i_y_s > i_neg1s) & (i_y_s < i_inp_H);
    auto nw_mask = cast<scalar_t>(must_in_bound ? i_neg1s : (w_mask & n_mask));
    auto ne_mask = cast<scalar_t>(e_mask & n_mask);
    auto sw_mask = cast<scalar_t>(w_mask & s_mask);
    auto se_mask = cast<scalar_t>(e_mask & s_mask);

    return std::make_tuple(
      n, s, w, e,
      nw, ne, sw, se,
      nw_mask, ne_mask, sw_mask, se_mask,
      i_y_n, i_x_w);
  }

  inline void forward(scalar_t *out_ptr, const scalar_t *inp_slice_ptr,
                      int64_t out_sC, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    auto x = compute_W.apply(grid_x);
    auto y = compute_H.apply(grid_y);

    auto interp_params = compute_interp_params(x, y);

    auto nw = std::get<4>(interp_params);
    auto ne = std::get<5>(interp_params);
    auto sw = std::get<6>(interp_params);
    auto se = std::get<7>(interp_params);

    auto nw_mask = std::get<8>(interp_params);
    auto ne_mask = std::get<9>(interp_params);
    auto sw_mask = std::get<10>(interp_params);
    auto se_mask = std::get<11>(interp_params);

    auto i_y_n = std::get<12>(interp_params);
    auto i_x_w = std::get<13>(interp_params);

    auto i_nw_offset = i_y_n * i_inp_sH + i_x_w * i_inp_sW;
    auto i_ne_offset = i_nw_offset + i_inp_sW;
    auto i_sw_offset = i_nw_offset + i_inp_sH;
    auto i_se_offset = i_sw_offset + i_inp_sW;

    #pragma unroll
    for (int64_t c = 0; c < C; ++c, out_ptr += out_sC, inp_slice_ptr += inp_sC) {
      // mask_gather zeros out the mask, so we need to make copies
      Vec nw_mask_copy = nw_mask;
      Vec ne_mask_copy = ne_mask;
      Vec sw_mask_copy = sw_mask;
      Vec se_mask_copy = se_mask;
      auto nw_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_nw_offset, nw_mask_copy);
      auto ne_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_ne_offset, ne_mask_copy);
      auto sw_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_sw_offset, sw_mask_copy);
      auto se_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_se_offset, se_mask_copy);
      auto interpolated = (nw_val * nw) + (ne_val * ne) + (sw_val * sw) + (se_val * se);
      interpolated.store(static_cast<void*>(out_ptr), len);
    }
  }

  inline void backward(scalar_t *gInp_slice_ptr, scalar_t *gGrid_ptr,
                       const scalar_t *gOut_ptr, const scalar_t *inp_slice_ptr,
                       int64_t gInp_sC, int64_t gOut_sC,
                       const Vec& grid_x, const Vec& grid_y, int64_t len) const {
    Vec x, y, gx_mult, gy_mult;
    std::tie(x, gx_mult) = compute_W.apply_get_grad(grid_x);
    std::tie(y, gy_mult) = compute_H.apply_get_grad(grid_y);

    Vec n, s, w, e, nw, ne, sw, se, nw_mask, ne_mask, sw_mask, se_mask;
    iVec i_y_n, i_x_w;

    std::tie(
      n, s, w, e, nw, ne, sw, se, nw_mask, ne_mask, sw_mask, se_mask,
      i_y_n, i_x_w) = compute_interp_params(x, y);

    auto i_nw_offset = i_y_n * i_inp_sH + i_x_w * i_inp_sW;
    auto i_ne_offset = i_nw_offset + i_inp_sW;
    auto i_sw_offset = i_nw_offset + i_inp_sH;
    auto i_se_offset = i_sw_offset + i_inp_sW;

    auto i_gInp_nw_offset = i_y_n * i_inp_W + i_x_w;
    auto i_gInp_ne_offset = i_gInp_nw_offset + i_ones;
    auto i_gInp_sw_offset = i_gInp_nw_offset + i_inp_W;
    auto i_gInp_se_offset = i_gInp_sw_offset + i_ones;

    integer_t i_gInp_nw_offset_arr[iVec::size];
    integer_t i_gInp_ne_offset_arr[iVec::size];
    integer_t i_gInp_sw_offset_arr[iVec::size];
    integer_t i_gInp_se_offset_arr[iVec::size];
    i_gInp_nw_offset.store(i_gInp_nw_offset_arr);
    i_gInp_ne_offset.store(i_gInp_ne_offset_arr);
    i_gInp_sw_offset.store(i_gInp_sw_offset_arr);
    i_gInp_se_offset.store(i_gInp_se_offset_arr);

    integer_t i_nw_mask_arr[iVec::size];
    integer_t i_ne_mask_arr[iVec::size];
    integer_t i_sw_mask_arr[iVec::size];
    integer_t i_se_mask_arr[iVec::size];
    nw_mask.store(i_nw_mask_arr);
    ne_mask.store(i_ne_mask_arr);
    sw_mask.store(i_sw_mask_arr);
    se_mask.store(i_se_mask_arr);

    scalar_t gInp_corner_arr[Vec::size];

    auto gx = zeros, gy = zeros;
    #pragma unroll
    for (int64_t c = 0; c < C; ++c, gOut_ptr += gOut_sC, gInp_slice_ptr += gInp_sC, inp_slice_ptr += inp_sC) {
      auto gOut = Vec::loadu(gOut_ptr, len);
      (nw * gOut).store(gInp_corner_arr);
      mask_scatter_add(gInp_corner_arr, gInp_slice_ptr, i_gInp_nw_offset_arr, i_nw_mask_arr, len);
      (ne * gOut).store(gInp_corner_arr);
      mask_scatter_add(gInp_corner_arr, gInp_slice_ptr, i_gInp_ne_offset_arr, i_ne_mask_arr, len);
      (sw * gOut).store(gInp_corner_arr);
      mask_scatter_add(gInp_corner_arr, gInp_slice_ptr, i_gInp_sw_offset_arr, i_sw_mask_arr, len);
      (se * gOut).store(gInp_corner_arr);
      mask_scatter_add(gInp_corner_arr, gInp_slice_ptr, i_gInp_se_offset_arr, i_se_mask_arr, len);

      // mask_gather zeros out the mask, so we need to make copies
      Vec nw_mask_copy = nw_mask;
      Vec ne_mask_copy = ne_mask;
      Vec sw_mask_copy = sw_mask;
      Vec se_mask_copy = se_mask;
      auto nw_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_nw_offset, nw_mask_copy);
      auto ne_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_ne_offset, ne_mask_copy);
      auto sw_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_sw_offset, sw_mask_copy);
      auto se_val = mask_gather<sizeof(scalar_t)>(zeros, inp_slice_ptr, i_se_offset, se_mask_copy);

      gx = gx + ((ne_val - nw_val) * s + (se_val - sw_val) * n) * gOut;
      gy = gy + ((sw_val - nw_val) * e + (se_val - ne_val) * w) * gOut;
    }

    gx = gx * gx_mult;
    gy = gy * gy_mult;

    constexpr int64_t step = Vec::size;
    auto interleaved_gGrid = interleave2(gx, gy);
    std::get<0>(interleaved_gGrid).store(gGrid_ptr,
                                         std::min(len * 2, step));
    std::get<1>(interleaved_gGrid).store(gGrid_ptr + step,
                                         std::max(static_cast<int64_t>(0), len * 2 - step));
  }
};

template<typename scalar_t, GridSamplerPadding padding>
struct ApplyGridSample<scalar_t, 2, padding, GridSamplerInterpolation::Nearest> {
  using Vec = Vec256<scalar_t>;
  using integer_t = int_same_size_t<scalar_t>;
  using iVec = Vec256<integer_t>;

  const iVec i_inp_H;
  const iVec i_inp_W;
  const iVec i_inp_sH;
  const iVec i_inp_sW;
  const iVec i_neg1s = iVec(-1);
  const iVec i_ones = iVec(-1);
  const Vec ones = Vec(1);
  const Vec zeros = Vec(0);
  int64_t C;
  int64_t inp_sC;
  const ComputeLocation<scalar_t, padding> compute_H;
  const ComputeLocation<scalar_t, padding> compute_W;
  const bool must_in_bound = padding != GridSamplerPadding::Zeros;

  ApplyGridSample(const TensorAccessor<scalar_t, 4>& input)
    : i_inp_H(iVec(input.size(2)))
    , i_inp_W(iVec(input.size(3)))
    , i_inp_sH(input.stride(2))
    , i_inp_sW(input.stride(3))
    , C(input.size(1))
    , inp_sC(input.stride(1))
    , compute_H(input.size(2))
    , compute_W(input.size(3)) {}

  inline void forward(scalar_t *out_ptr, const scalar_t *inp_slice_ptr,
                      int64_t out_sC, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    auto x = compute_W.apply(grid_x);
    auto y = compute_H.apply(grid_y);

    auto x_nearest = x.round();
    auto y_nearest = y.round();

    auto i_x_nearest = convert_to_int_of_same_size(x_nearest);
    auto i_y_nearest = convert_to_int_of_same_size(y_nearest);

    auto i_mask = must_in_bound ? i_neg1s
                                : (i_x_nearest > i_neg1s) & (i_x_nearest < i_inp_W) &
                                  (i_y_nearest > i_neg1s) & (i_y_nearest < i_inp_H);
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

  inline void backward(scalar_t *gInp_slice_ptr, scalar_t *gGrid_ptr,
                       const scalar_t *gOut_ptr, const scalar_t *inp_slice_ptr,
                       int64_t gInp_sC, int64_t gOut_sC,
                       const Vec& grid_x, const Vec& grid_y, int64_t len) const {
    auto x = compute_W.apply(grid_x);
    auto y = compute_H.apply(grid_y);

    auto x_nearest = x.round();
    auto y_nearest = y.round();

    auto i_x_nearest = convert_to_int_of_same_size(x_nearest);
    auto i_y_nearest = convert_to_int_of_same_size(y_nearest);

    auto i_mask = must_in_bound ? i_neg1s
                                : (i_x_nearest > i_neg1s) & (i_x_nearest < i_inp_W) &
                                  (i_y_nearest > i_neg1s) & (i_y_nearest < i_inp_H);

    auto i_gInp_offset = i_y_nearest * i_inp_W + i_x_nearest;  // gInp is contiguous

    integer_t mask_arr[iVec::size];
    i_mask.store(mask_arr);
    integer_t gInp_offset_arr[iVec::size];
    i_gInp_offset.store(gInp_offset_arr);

    #pragma unroll
    for (int64_t c = 0; c < C; ++c, gOut_ptr += gOut_sC, gInp_slice_ptr += gInp_sC) {
      mask_scatter_add(gOut_ptr, gInp_slice_ptr, gInp_offset_arr, mask_arr, len);
    }

    // grid has zero 0 gradient in Nearest mode
    std::memset(gGrid_ptr, 0, sizeof(scalar_t) * len * 2);
  }
};

template<typename scalar_t, typename ApplyFn>
static inline void grid_sample_2d_grid_slice_iterator(
    const TensorAccessor<scalar_t, 3>& grid_slice, const ApplyFn &apply_fn) {
  int64_t out_H = grid_slice.size(0);
  int64_t out_W = grid_slice.size(1);
  int64_t grid_sH = grid_slice.stride(0);
  int64_t grid_sW = grid_slice.stride(1);
  int64_t grid_sCoor = grid_slice.stride(2);
  auto grid_ptr = grid_slice.data();

  using Vec = Vec256<scalar_t>;
  using iVec = Vec256<int_same_size_t<scalar_t>>;
  constexpr int64_t step = Vec::size;

  // Loop over each output pixel:
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
    auto zeros = Vec(0);
    auto total_size = out_H * out_W;
    for (int64_t spatial_offset = 0; spatial_offset < total_size; spatial_offset += step) {
      auto grid_offset = spatial_offset * 2;
      auto len = std::min(step, total_size - spatial_offset);
      auto vec1 = Vec::loadu(grid_ptr + grid_offset,
                             std::min(step, len * 2));
      auto vec2 = Vec::loadu(grid_ptr + grid_offset + step,
                             std::max(static_cast<int64_t>(0), len * 2 - step));
      auto vec_xy_pair = deinterleave2(vec1, vec2);

      auto x = std::get<0>(vec_xy_pair);
      auto y = std::get<1>(vec_xy_pair);

      // make sure that x and y are valid grid sample locations
      if (len < step) {
        x = Vec::set(zeros, x, len);
        y = Vec::set(zeros, y, len);
      }
      apply_fn(x, y, spatial_offset, len);
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
    auto zeros = Vec(0);
    auto line_fn = [&](const scalar_t *grid_ptr_x, const scalar_t *grid_ptr_y,
                       int64_t out_base_offset, int64_t total_size) {
      for (int64_t i = 0; i < total_size; i += step) {
        auto len = std::min(step, total_size - i);
        auto x = Vec::loadu(grid_ptr_x + i, len);
        auto y = Vec::loadu(grid_ptr_y + i, len);
        // make sure that x and y are valid grid sample locations
        if (len < step) {
          x = Vec::set(zeros, x, len);
          y = Vec::set(zeros, y, len);
        }
        apply_fn(x, y, out_base_offset + i, len);
      }
    };

    if (at::geometry_is_contiguous({out_H, out_W}, {grid_sH, grid_sW})) {
      // If [H, W] is contiguous, apply line_fn once.
      line_fn(grid_ptr, grid_ptr + grid_sCoor, 0, out_H * out_W);
    } else {
      // If only [W] is contiguous, apply line_fn once for each h slice.
      auto grid_ptr_NH = grid_ptr;
      for (int64_t h = 0; h < out_H; h++) {
        line_fn(grid_ptr_NH, grid_ptr_NH + grid_sCoor, h * out_W, out_W);
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
      auto grid_ptr_x = grid_ptr + h * grid_sH;
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

template<typename scalar_t,
         GridSamplerInterpolation interp_mode,
         GridSamplerPadding padding_mode>
static inline
void grid_sample_2d_vec_kernel(Tensor& output, const Tensor& input, const Tensor& grid) {
  auto out_acc = output.accessor<scalar_t, 4>();
  auto inp_acc = input.accessor<scalar_t, 4>();
  auto grid_acc = grid.accessor<scalar_t, 4>();

  ApplyGridSample<scalar_t, 2, padding_mode, interp_mode> grid_sample(inp_acc);

  auto out_sC = out_acc.stride(1);

  #ifdef _OPENMP
  #pragma parallel for if (N > 1)
  #endif
  for (int64_t n = 0; n < grid_acc.size(0); ++n) {
    auto out_slice_ptr = out_acc[n].data();
    auto inp_slice_ptr = inp_acc[n].data();
    grid_sample_2d_grid_slice_iterator(
      grid_acc[n],
      [&](const Vec256<scalar_t>& grid_x, const Vec256<scalar_t>& grid_y,
          int64_t spatial_offset, int64_t len) {
        grid_sample.forward(out_slice_ptr + spatial_offset, inp_slice_ptr,
                            out_sC, grid_x, grid_y, len);
      });
  }
}

template<typename scalar_t,
         GridSamplerInterpolation interp_mode,
         GridSamplerPadding padding_mode>
static inline
void grid_sample_2d_backward_vec_kernel(Tensor& grad_input, Tensor& grad_grid,
                                        const Tensor& input, const Tensor& grid,
                                        const Tensor& grad_output_) {

  // grad_output should be contiguous most of time. Ensuring that it is
  // contiguous can greatly simplify this code.
  auto grad_output = grad_output_.contiguous();

  auto gInp_acc = grad_input.accessor<scalar_t, 4>();
  auto gGrid_acc = grad_grid.accessor<scalar_t, 4>();
  auto inp_acc = input.accessor<scalar_t, 4>();
  auto grid_acc = grid.accessor<scalar_t, 4>();
  auto gOut_acc = grad_output.accessor<scalar_t, 4>();

  ApplyGridSample<scalar_t, 2, padding_mode, interp_mode> grid_sample(inp_acc);

  auto gOut_sC = gOut_acc.stride(1);
  auto gInp_sC = gInp_acc.stride(1);

  #ifdef _OPENMP
  #pragma parallel for if (N > 1)
  #endif
  for (int64_t n = 0; n < grid_acc.size(0); ++n) {
    auto gInp_slice_ptr = gInp_acc[n].data();
    auto gGrid_slice_ptr = gGrid_acc[n].data();
    auto inp_slice_ptr = inp_acc[n].data();
    auto gOut_slice_ptr = gOut_acc[n].data();
    grid_sample_2d_grid_slice_iterator(
      grid_acc[n],
      [&](const Vec256<scalar_t>& grid_x, const Vec256<scalar_t>& grid_y,
          int64_t spatial_offset, int64_t len) {

        grid_sample.backward(gInp_slice_ptr, gGrid_slice_ptr + 2 * spatial_offset,
                             gOut_slice_ptr + spatial_offset, inp_slice_ptr,
                             gInp_sC, gOut_sC, grid_x, grid_y, len);
      });
  }
}

Tensor grid_sampler_2d_cpu_kernel_impl(const Tensor& input, const Tensor& grid,
                                       int64_t interpolation_mode,
                                       int64_t padding_mode) {
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

std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cpu_kernel_impl(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                                         int64_t interpolation_mode,
                                         int64_t padding_mode) {
  auto grad_input = at::zeros_like(input);
  auto grad_grid = at::empty_like(grid);

#define HANDLE_CASE(interp, padding)                                            \
  case padding: {                                                               \
    grid_sample_2d_backward_vec_kernel<scalar_t, interp, padding>(grad_input, grad_grid, input, grid, grad_output);  \
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

  return std::make_tuple(grad_input, grad_grid);
}

}

REGISTER_DISPATCH(grid_sampler_2d_cpu_kernel, &grid_sampler_2d_cpu_kernel_impl);
REGISTER_DISPATCH(grid_sampler_2d_backward_cpu_kernel, &grid_sampler_2d_backward_cpu_kernel_impl);


}}  // namespace at::native

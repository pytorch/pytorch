#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/GridSampler.h>
#include <ATen/native/cpu/GridSamplerKernel.h>
#include <ATen/cpu/vml.h>
#include <c10/util/C++17.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace at { namespace native { namespace {

/**  NOTE [ Grid Sample CPU Kernels ]
 *
 *   Implementation of vectorized grid sample CPU kernels is divided into three
 *   parts. More detailed description exist after this paragraph, but on a high
 *   level, they are
 *   1. `ComputeLocation` struct
 *      + Computes the interpolation location basing on padding mode.
 *   2. `ApplyGridSample` struct
 *      + Owns N (# spatial dims) `ComputeLocation` structs, and uses them to
 *        compute the interpolation locations.
 *      + Interpolates the values and writes to output.
 *   3. `grid_sample_2d_grid_slice_iterator` function
 *      + Iterates over a slice of the grid tensor based on the geometry by the
 *        spatial ordering, i.e., the first iteration will process grid values
 *           grid[n, 0, 0, :], grid[n, 0, 1, :], grid[n, 0, 2, :], ...
 *        (Recall that, e.g., 2D grid has shape [N x H x W x 2], so grid[n, ...]
 *         is a slice, and grid[n, h, w, :] contains the values for a single
 *         output spatial location.)
 *      + Applies a given operator at each iteration, so we can use the same
 *        pattern for forward and backward.
 *
 *   Putting everything together, we have, e.g., the forward kernel implemented
 *   as
 *
 *      // `ApplyGridSample` struct that processes grid values, extracts and
 *      // interpolates input values, and write to output.
 *      ApplyGridSample<scalar_t, 2, interp, padding> grid_sample(input_accessor);
 *
 *      // For each slice, we call `grid_sample_2d_grid_slice_iterator` with
 *      //   1. the grid slice, and
 *      //   2. a lambda that takes in
 *      //      i.   location vectors (x and y for 2D) extracted from grid
 *      //      ii.  `spatial_offset` as the spatial offset of these vectors
 *      //           from the beginning of this slice.
 *      //      iii. `len` as the number of valid locations in the vectors.
 *      //           (There might not be enough near boundary.)
 *      for (const auto n : c10::irange(input_accessor.size(0))) {
 *        grid_sample_2d_grid_slice_iterator(
 *          grid_accessor[n],
 *          [&](const Vectorized<scalar_t>& grid_x,
 *              const Vectorized<scalar_t>& grid_y,
 *              int64_t spatial_offset, int64_t len) {
 *            grid_sample.forward(out_accessor[n], input_accessor[n],
 *                                spatial_offset, grid_x, grid_y, len);
 *          });
 *      }
 *
 *   Now we talk about details of each of these three parts:
 *
 *   1. `ComputeLocation` struct
 *      Transforms grid values into interpolation locations of the input tensor
 *      for a particular spatial dimension, based on the size of that dimension
 *      in input tensor, and the padding mode.
 *
 *        template<typename scalar_t, GridSamplerPadding padding>
 *        struct ComputeLocation {
 *          using Vec = Vectorized<scalar_t>;
 *
 *          // ctor
 *          ComputeLocation(int64_t size);
 *
 *          // Given grid values `in`, return the interpolation locations after
 *          // un-normalization and padding mechanism (elementwise).
 *          Vec apply(const Vec &in) const;
 *
 *          // Similar to `apply`, but also returns `d apply(in) / d in`
 *          // (elementwise).
 *          // this is often used in gradient computation.
 *          std::pair<Vec, Vec> apply_get_grad(const Vec &in) const;
 *        };
 *
 *   2. `ApplyGridSample` struct
 *      Owns N `ComputeLocation` structs, where N is the number of spatial
 *      dimensions. Given N input grid vectors (one for each spatial dimension)
 *      and spatial offset, it gets the interpolation locations from
 *      `ComputeLocation`s, applies interpolation procedure, and then writes to
 *      the output (or grad_input & grad_grid in backward).
 *
 *        template<typename scalar_t, int spatial_dim,
 *                 GridSamplerInterpolation interp,
 *                 GridSamplerPadding padding>
 *        struct ApplyGridSample {
 *
 *          // ctor
 *          ApplyGridSample(const TensorAccessor<scalar_t, 4>& input);
 *
 *          // Applies grid sampling (forward) procedure:
 *          //   1. computes interpolation locations from grid values `grid_x`
 *          //      and `grid_y`,
 *          //   2. interpolates output values using the locations and input
 *          //      data in `inp_slice`, and
 *          //   3. writes the first `len` values in the interpolated vector to
 *          //      `out_slice` with spatial offset being `offset`.
 *          //
 *          // This assimes that `grid_x` and `grid_y` all contain valid grid
 *          // values \in [-1, 1], even at indices greater than `len`.
 *          //
 *          // The `*_slice` argument names mean samples within a batch (i.e.,
 *          // with the batch dimension sliced out).
 *          void forward(TensorAccessor<scalar_t, 3>& out_slice,
 *                       const TensorAccessor<scalar_t, 3>& inp_slice,
 *                       int64_t offset, const Vec& grid_x, const Vec& grid_y,
 *                       int64_t len) const;
 *
 *          // Applies grid sampling (backward) procedure. Arguments semantics
 *          // and strategy are similar to those of `forward`, with the
 *          // exception that `backward` has branches based on whether `input`
 *          // requires gradient (passed in as a template parameter). The
 *          // TensorAccessor for the input gradient is also given as a
 *          // pointer instead of reference, so that it can be null if the
 *          // gradient is not calculated.
 *          template <bool input_requires_grad>
 *          void backward(TensorAccessor<scalar_t, 3>* gInp_slice_ptr,
 *                        TensorAccessor<scalar_t, 3>& gGrid_slice,
 *                        const TensorAccessor<scalar_t, 3>& gOut_slice,
 *                        const TensorAccessor<scalar_t, 3>& inp_slice,
 *                        int64_t offset, const Vec& grid_x, const Vec& grid_y,
 *                        int64_t len) const;
 *        };
 *
 *   3. `grid_sample_2d_grid_slice_iterator` function
 *      Among the tensors we work with, we know that the output tensors are
 *      contiguous (i.e., `output` in forward, and `grad_input` & `grad_grid` in
 *      backward), we need to randomly read `input` anyways, and `grad_output`
 *      usually comes from autograd and is often contiguous. So we base our
 *      iterating strategy on the geometry of grid.
 *      `grid_sample_2d_grid_slice_iterator` function provides an abstraction to
 *      efficiently iterates through a `grid` slice (without batch dimension).
 *      See comments of that function on the specific cases and strategies used.
 *
 *        template<typename scalar_t, typename ApplyFn>
 *        void grid_sample_2d_grid_slice_iterator(
 *          const TensorAccessor<scalar_t, 3>& grid_slice,
 *          const ApplyFn &apply_fn);
 *
 *      `apply_fn` is a function/lambda that takes in
 *           i.   location vectors (x and y for 2D) extracted from grid
 *           ii.  `spatial_offset` as the spatial offset of these vectors
 *                from the beginning of this slice.
 *           iii. `len` as the number of valid locations in the vectors.
 *                (There might not be enough near boundary.)

 *       It should be callable as if it has declaration:
 *          void apply_fn(const Vectorized<scalar_t>& grid_x,
 *                        const Vectorized<scalar_t>& grid_y,
 *                        int64_t spatial_offset, int64_t len);
 *
 *      `apply_fn` will be called multiple times, and together cover the entire
 *      output spatial space.
 *
 *  Now you should be able to understand everything about the implementation of
 *  2D forward kernel shown at the beginning of this note.
 *
 **/


using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;
using namespace at::vec;


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ComputeLocation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Struct to compute interpolation location from grid values, and to apply
// padding mechanism (e.g., reflection).
// See NOTE [ Grid Sample CPU Kernels ] for details.

template<typename scalar_t, bool align_corners>
struct ComputeLocationBase;

template<typename scalar_t>
struct ComputeLocationBase<scalar_t, /*align_corners=*/true> {
  using Vec = Vectorized<scalar_t>;

  // values are clipped to between 0 and max_val
  const scalar_t max_val;
  // unnormalization scaling factor
  const scalar_t scaling_factor;
  // reflection parameters: reflected coordinates land in [low, low+span] inclusive
  const scalar_t low; // only used when align_corners=False
  const scalar_t twice_span;
  // if the reflecting span is empty, all reflected coords are set to 0
  const bool empty;

  ComputeLocationBase(int64_t size)
    : max_val(static_cast<scalar_t>(size - 1))
    , scaling_factor(static_cast<scalar_t>(size - 1) / 2)
    , low(static_cast<scalar_t>(0))
    , twice_span(static_cast<scalar_t>(size - 1) * 2)
    , empty(size <= 1) {}

  inline Vec unnormalize(const Vec &in) const {
    return (in + Vec(1)) * Vec(scaling_factor);
  }

  inline Vec clip_coordinates(const Vec &in) const {
    // Invert order of clamp_min operands in order to clamp Nans to zero
    return clamp_max(Vec(max_val), clamp_min(Vec(0), in));
  }

  // same as clip_coordinates but also returns the gradient multiplier
  inline std::pair<Vec, Vec> clip_coordinates_get_grad(const Vec &in) const {
    using int_t = int_same_size_t<scalar_t>;
    auto bounded_lo = maximum(in, Vec(0));
    // Integral type equality comparison is very very fast because it just looks
    // at the bits. Casting is free too. So we use the following pattern instead
    // of comparison + blendv.
    // Note that it is important for the gradient calculation that borders
    // are considered out of bounds.
    auto in_bound_lo = cast<scalar_t>(cast<int_t>(bounded_lo) != cast<int_t>(Vec(0)));
    auto res = minimum(bounded_lo, Vec(max_val));
    auto in_bound_hi = cast<scalar_t>(cast<int_t>(res) != cast<int_t>(Vec(max_val)));
    return std::make_pair(res, in_bound_lo & in_bound_hi);
  }

  inline Vec reflect_coordinates(const Vec &in) const {
    if (empty) {
      return Vec(0);
    }
    Vec twice_span_vec(twice_span);
    auto abs_in = in.abs();
    auto fdouble_flips = abs_in / twice_span_vec;
    auto double_flips = fdouble_flips.trunc();
    auto extra = abs_in - double_flips * twice_span_vec;
    // Now we need to test if extra > max_val to find out if another flip is
    // needed. The following comparison does that and returns the correct
    // flipped value.
    return minimum(extra, twice_span_vec - extra);
  }

  // same as reflect_coordinates but also returns the gradient multiplier
  inline std::pair<Vec, Vec> reflect_coordinates_get_grad(const Vec &in) const {
    if (empty) {
      return std::make_pair(Vec(0), Vec(0));
    }
    Vec twice_span_vec(twice_span);
    auto neg_in = in < Vec(0);
    auto abs_in = in.abs();
    auto fdouble_flips = abs_in / twice_span_vec;
    auto double_flips = fdouble_flips.trunc();

    auto extra = abs_in - double_flips * twice_span_vec;
    auto reflected_extra = twice_span_vec - extra;
    auto one_more_flip = extra > reflected_extra;

    return std::make_pair(
      Vec::blendv(extra, reflected_extra, one_more_flip),
      Vec::blendv(Vec(1), Vec(-1), one_more_flip ^ neg_in)
    );
  }
};

template<typename scalar_t>
struct ComputeLocationBase<scalar_t, /*align_corners=*/false> {
  using Vec = Vectorized<scalar_t>;

  // values are clipped to between 0 and max_val
  const scalar_t max_val;
  // unnormalization scaling factor
  const scalar_t scaling_factor;
  // reflection parameters: reflected coordinates land in [low, low+span] inclusive
  const scalar_t low;
  const scalar_t twice_span;
  // if the reflecting span is empty, all reflected coords are set to 0
  const bool empty; // only used when align_corners=True

  ComputeLocationBase(int64_t size)
    : max_val(static_cast<scalar_t>(size - 1))
    , scaling_factor(static_cast<scalar_t>(size) / 2)
    , low(static_cast<scalar_t>(-0.5))
    , twice_span(static_cast<scalar_t>(size) * 2)
    , empty(size <= 0) {}

  inline Vec unnormalize(const Vec &in) const {
    return (in + Vec(1)) * Vec(scaling_factor) - Vec(0.5);
  }

  inline Vec clip_coordinates(const Vec &in) const {
    // Invert order of clamp_min operands in order to clamp Nans to zero
    return clamp_max(Vec(max_val), clamp_min(Vec(0), in));
  }

  // same as clip_coordinates but also returns the gradient multiplier
  inline std::pair<Vec, Vec> clip_coordinates_get_grad(const Vec &in) const {
    using int_t = int_same_size_t<scalar_t>;
    auto bounded_lo = maximum(in, Vec(0));
    // Integral type equality comparison is very very fast because it just looks
    // at the bits. Casting is free too. So we use the following pattern instead
    // of comparison + blendv.
    // Note that it is important for the gradient calculation that borders
    // are considered out of bounds.
    auto in_bound_lo = cast<scalar_t>(cast<int_t>(bounded_lo) != cast<int_t>(Vec(0)));
    auto res = minimum(bounded_lo, Vec(max_val));
    auto in_bound_hi = cast<scalar_t>(cast<int_t>(res) != cast<int_t>(Vec(max_val)));
    return std::make_pair(res, in_bound_lo & in_bound_hi);
  }

  inline Vec reflect_coordinates(const Vec &in) const {
    Vec twice_span_vec(twice_span), low_vec(low);
    // Since reflection is around low and low+span, subtract low before
    // the reflection, and then add it back at the end.
    auto abs_in = (in - low_vec).abs();
    auto fdouble_flips = abs_in / twice_span_vec;
    auto double_flips = fdouble_flips.trunc();
    auto extra = abs_in - double_flips * twice_span_vec;
    // Now we need to test if extra > max_val to find out if another flip is
    // needed. The following comparison does that and returns the correct
    // flipped value.
    return minimum(extra, twice_span_vec - extra) + low_vec;
  }

  // same as reflect_coordinates but also returns the gradient multiplier
  inline std::pair<Vec, Vec> reflect_coordinates_get_grad(const Vec &in) const {
    Vec twice_span_vec(twice_span), low_vec(low);
    Vec in_minus_low = in - low_vec;
    auto neg_in = in_minus_low < Vec(0);
    auto abs_in = in_minus_low.abs();
    auto fdouble_flips = abs_in / twice_span_vec;
    auto double_flips = fdouble_flips.trunc();

    auto extra = abs_in - double_flips * twice_span_vec;
    auto reflected_extra = twice_span_vec - extra;
    auto one_more_flip = extra > reflected_extra;

    return std::make_pair(
      Vec::blendv(extra, reflected_extra, one_more_flip) + low_vec,
      Vec::blendv(Vec(1), Vec(-1), one_more_flip ^ neg_in)
    );
  }
};

template<typename scalar_t, GridSamplerPadding padding, bool align_corners>
struct ComputeLocation;

template<typename scalar_t, bool align_corners>
struct ComputeLocation<scalar_t, GridSamplerPadding::Zeros, align_corners>
  : ComputeLocationBase<scalar_t, align_corners> {
  using Vec = Vectorized<scalar_t>;
  using ComputeLocationBase<scalar_t, align_corners>::unnormalize;
  using ComputeLocationBase<scalar_t, align_corners>::scaling_factor;

  using ComputeLocationBase<scalar_t, align_corners>::ComputeLocationBase;

  inline Vec apply(const Vec &in) const {
    return unnormalize(in);
  }

  inline Vec compute_coordinates(const Vec &in) const {
    return in;
  }

  inline std::pair<Vec, Vec> apply_get_grad(const Vec &in) const {
    return std::make_pair(unnormalize(in), Vec(scaling_factor));
  }
};

template<typename scalar_t, bool align_corners>
struct ComputeLocation<scalar_t, GridSamplerPadding::Border, align_corners>
  : ComputeLocationBase<scalar_t, align_corners> {
  using Vec = Vectorized<scalar_t>;
  using ComputeLocationBase<scalar_t, align_corners>::unnormalize;
  using ComputeLocationBase<scalar_t, align_corners>::clip_coordinates;
  using ComputeLocationBase<scalar_t, align_corners>::clip_coordinates_get_grad;
  using ComputeLocationBase<scalar_t, align_corners>::scaling_factor;

  using ComputeLocationBase<scalar_t, align_corners>::ComputeLocationBase;

  inline Vec apply(const Vec &in) const {
    return clip_coordinates(unnormalize(in));
  }

  inline Vec compute_coordinates(const Vec &in) const {
    return clip_coordinates(in);
  }

  inline std::pair<Vec, Vec> apply_get_grad(const Vec &in) const {
    Vec res, grad_clip;
    std::tie(res, grad_clip) = clip_coordinates_get_grad(unnormalize(in));
    return std::make_pair(res, grad_clip & Vec(scaling_factor));
  }
};

template<typename scalar_t, bool align_corners>
struct ComputeLocation<scalar_t, GridSamplerPadding::Reflection, align_corners>
  : ComputeLocationBase<scalar_t, align_corners> {
  using Vec = Vectorized<scalar_t>;
  using ComputeLocationBase<scalar_t, align_corners>::unnormalize;
  using ComputeLocationBase<scalar_t, align_corners>::clip_coordinates;
  using ComputeLocationBase<scalar_t, align_corners>::clip_coordinates_get_grad;
  using ComputeLocationBase<scalar_t, align_corners>::reflect_coordinates;
  using ComputeLocationBase<scalar_t, align_corners>::reflect_coordinates_get_grad;
  using ComputeLocationBase<scalar_t, align_corners>::scaling_factor;

  using ComputeLocationBase<scalar_t, align_corners>::ComputeLocationBase;

  inline Vec apply(const Vec &in) const {
    auto res = reflect_coordinates(unnormalize(in));
    res = clip_coordinates(res);
    return res;
  }

  inline Vec compute_coordinates(const Vec &in) const {
    auto res = reflect_coordinates(in);
    res = clip_coordinates(res);
    return res;
  }

  inline std::pair<Vec, Vec> apply_get_grad(const Vec &in) const {
    Vec res, grad_refl, grad_clip, grad(scaling_factor);
    std::tie(res, grad_refl) = reflect_coordinates_get_grad(unnormalize(in));
    grad = grad_refl * grad;
    std::tie(res, grad_clip) = clip_coordinates_get_grad(res);
    grad = grad_clip & grad;
    return std::make_pair(res, grad);
  }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ApplyGridSample ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Struct to apply grid sample (reading from input, interpolate, and write to
// output).
// See NOTE [ Grid Sample CPU Kernels ] for details.

inline std::tuple<Vectorized<float>, Vectorized<float>>
mask_gather_bfloat16(const Vectorized<float>& src0, const Vectorized<float>& src1, BFloat16 const* base_addr,
                   int32_t const* vindex, Vectorized<float>& mask0, Vectorized<float>& mask1) {
  using Vecf = Vectorized<float>;
  static constexpr int size = Vectorized<BFloat16>::size();
  float src_arr[size];
  int32_t mask_arr[size];  // use int type so we can logical and
  src0.store(static_cast<void*>(src_arr));
  src1.store(static_cast<void*>(src_arr + size/2));
  mask0.store(static_cast<void*>(mask_arr));
  mask1.store(static_cast<void*>(mask_arr + size/2));
  // vindex.store(static_cast<void*>(index_arr));
  float buffer[size];
  for (int64_t i = 0; i < size; i++) {
    if (mask_arr[i] & 0x01) {  // check highest bit
      buffer[i] = base_addr[vindex[i]];
    } else {
      buffer[i] = src_arr[i];
    }
  }
  mask0 = Vecf();  // "zero out" mask
  mask1 = Vecf();  // "zero out" mask
  return std::make_tuple(Vecf::loadu(static_cast<void*>(buffer)), Vecf::loadu(static_cast<void*>(buffer+ size/2)));
}

template<typename scalar_t>
static inline typename std::enable_if_t<!std::is_same<scalar_t, BFloat16>::value>
mask_scatter_add(const scalar_t *src, scalar_t* base_addr,
                 const int_same_size_t<scalar_t> *offsets,
                 const int_same_size_t<scalar_t> *mask, int64_t len) {
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (const auto i : c10::irange(len)) {
    if (mask[i] & 0x01) {
      base_addr[offsets[i]] += src[i];
    }
  }
}

static inline void
mask_scatter_add(const BFloat16 *src, BFloat16* base_addr,
                 const int32_t *offsets,
                 const int32_t *mask, int64_t len) {
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (int64_t i = 0; i < len; i++) {
    if (mask[i] & 0x01) {
      base_addr[offsets[i]] += src[i];
    }
  }
}

template<typename scalar_t, int spatial_dim,
         GridSamplerInterpolation interp,
         GridSamplerPadding padding,
         bool align_corners>
struct ApplyGridSample;

template<typename scalar_t, GridSamplerPadding padding, bool align_corners>
struct ApplyGridSample<scalar_t, 2, GridSamplerInterpolation::Bilinear,
                       padding, align_corners> {
  using Vec = Vectorized<scalar_t>;
  //use int_32t index for BFloat16
  using integer_t = typename std::conditional_t<std::is_same<scalar_t, BFloat16>::value, int32_t, int_same_size_t<scalar_t>>;
  using iVec = Vectorized<integer_t>;
  using scalar_tb = typename std::conditional_t<std::is_same<scalar_t, BFloat16>::value, float, scalar_t>;

  const int64_t inp_H;
  const int64_t inp_W;
  const int64_t inp_sH;
  const int64_t inp_sW;
  const int64_t C;
  const int64_t inp_sC;
  const ComputeLocation<scalar_tb, padding, align_corners> compute_H;
  const ComputeLocation<scalar_tb, padding, align_corners> compute_W;
  const bool must_in_bound = padding != GridSamplerPadding::Zeros;

  ApplyGridSample(const TensorAccessor<scalar_t, 4>& input)
    : inp_H(input.size(2))
    , inp_W(input.size(3))
    , inp_sH(input.stride(2))
    , inp_sW(input.stride(3))
    , C(input.size(1))
    , inp_sC(input.stride(1))
    , compute_H(input.size(2))
    , compute_W(input.size(3)
    ) {}

  template<typename Vec_t, typename iVec_t, typename scalar_type>
  inline std::tuple<
    Vec_t, Vec_t, Vec_t, Vec_t,       // distances to 4 sides
    Vec_t, Vec_t, Vec_t, Vec_t,       // interpolation weights wrt 4 corners
    Vec_t, Vec_t, Vec_t, Vec_t,       // in_bound masks
    iVec_t, iVec_t                // y_n and x_w
  >
  compute_interp_params(const Vec_t& x, const Vec_t& y) const {
    // get NE, NW, SE, SW pixel values from (x, y)
    // assuming we get exact integer representation and just use scalar_t
    // if we don't, the weights will be garbage anyways.
    auto x_w = x.floor();
    auto y_n = y.floor();

    // get distances to each side
    auto w = x - x_w;
    auto e = Vec_t(1) - w;
    auto n = y - y_n;
    auto s = Vec_t(1) - n;

    // get interpolation weights for each neighbor
    // e.g., for the nw corder, the weight is `dist_to_south * dist_to_east`.
    auto nw = s * e;
    auto ne = s * w;
    auto sw = n * e;
    auto se = n * w;

    auto i_x_w = convert_to_int_of_same_size(x_w);
    auto i_y_n = convert_to_int_of_same_size(y_n);
    auto i_x_e = i_x_w + iVec_t(1);
    auto i_y_s = i_y_n + iVec_t(1);

    // Use int comparison because it is much faster than float comp with AVX2
    // (latency 1 cyc vs. 4 cyc on skylake)
    // Avoid using the le and ge because those are not implemented in AVX2 and
    // are actually simulated using multiple instructions.
    auto w_mask = must_in_bound ? iVec_t(-1)  // true = all ones
                                : (i_x_w > iVec_t(-1)) & (i_x_w < iVec_t(inp_W));
    auto n_mask = must_in_bound ? iVec_t(-1)  // true = all ones
                                : (i_y_n > iVec_t(-1)) & (i_y_n < iVec_t(inp_H));
    auto e_mask = must_in_bound ? (i_x_e < iVec_t(inp_W))
                                : (i_x_e > iVec_t(-1)) & (i_x_e < iVec_t(inp_W));
    auto s_mask = must_in_bound ? (i_y_s < iVec_t(inp_H))
                                : (i_y_s > iVec_t(-1)) & (i_y_s < iVec_t(inp_H));
    auto nw_mask = cast<scalar_type>(must_in_bound ? iVec_t(-1) : (w_mask & n_mask));
    auto ne_mask = cast<scalar_type>(e_mask & n_mask);
    auto sw_mask = cast<scalar_type>(w_mask & s_mask);
    auto se_mask = cast<scalar_type>(e_mask & s_mask);

    return std::make_tuple(
      n, s, w, e,
      nw, ne, sw, se,
      nw_mask, ne_mask, sw_mask, se_mask,
      i_y_n, i_x_w);
  }

  template <typename scalar_type>
  inline typename std::enable_if_t<!std::is_same<scalar_type, BFloat16>::value>
  forward_bilinear(TensorAccessor<scalar_type, 3>& out_slice,
  const TensorAccessor<scalar_type, 3>& inp_slice,
  int64_t offset, const Vectorized<scalar_type>& grid_x, const Vectorized<scalar_type>& grid_y,
  int64_t len) const {
    auto x = compute_W.apply(grid_x);
    auto y = compute_H.apply(grid_y);

    auto interp_params = compute_interp_params<Vec, iVec, scalar_type>(x, y);

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

    auto i_nw_offset = i_y_n * iVec(inp_sH) + i_x_w * iVec(inp_sW);
    auto i_ne_offset = i_nw_offset + iVec(inp_sW);
    auto i_sw_offset = i_nw_offset + iVec(inp_sH);
    auto i_se_offset = i_sw_offset + iVec(inp_sW);

    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (const auto c : c10::irange(C)) {
      auto inp_slice_C_ptr = inp_slice[c].data();

      // mask_gather zeros out the mask, so we need to make copies
      Vec nw_mask_copy = nw_mask;
      Vec ne_mask_copy = ne_mask;
      Vec sw_mask_copy = sw_mask;
      Vec se_mask_copy = se_mask;
      auto nw_val = mask_gather<sizeof(scalar_type)>(Vec(0), inp_slice_C_ptr, i_nw_offset, nw_mask_copy);
      auto ne_val = mask_gather<sizeof(scalar_type)>(Vec(0), inp_slice_C_ptr, i_ne_offset, ne_mask_copy);
      auto sw_val = mask_gather<sizeof(scalar_type)>(Vec(0), inp_slice_C_ptr, i_sw_offset, sw_mask_copy);
      auto se_val = mask_gather<sizeof(scalar_type)>(Vec(0), inp_slice_C_ptr, i_se_offset, se_mask_copy);

      auto interpolated = (nw_val * nw) + (ne_val * ne) + (sw_val * sw) + (se_val * se);
      interpolated.store(out_slice[c].data() + offset, len);
    }
  }

  void forward_bilinear(TensorAccessor<BFloat16, 3>& out_slice,
                      const TensorAccessor<BFloat16, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len)  const {
    using Vecf = Vectorized<float>;
    Vecf grid_x0, grid_x1, grid_y0, grid_y1;
    std::tie(grid_x0, grid_x1) =  convert_bfloat16_float(grid_x);
    std::tie(grid_y0, grid_y1) =  convert_bfloat16_float(grid_y);

    auto x0 = compute_W.apply(grid_x0);
    auto y0 = compute_H.apply(grid_y0);
    auto x1 = compute_W.apply(grid_x1);
    auto y1 = compute_H.apply(grid_y1);

    auto interp_params = compute_interp_params<Vecf, iVec, float>(x0, y0);

    auto nw0 = std::get<4>(interp_params);
    auto ne0 = std::get<5>(interp_params);
    auto sw0 = std::get<6>(interp_params);
    auto se0 = std::get<7>(interp_params);

    auto nw_mask0 = std::get<8>(interp_params);
    auto ne_mask0 = std::get<9>(interp_params);
    auto sw_mask0 = std::get<10>(interp_params);
    auto se_mask0 = std::get<11>(interp_params);

    auto i_y_n = std::get<12>(interp_params);
    auto i_x_w = std::get<13>(interp_params);

    auto i_nw_offset0 = i_y_n * iVec(inp_sH) + i_x_w * iVec(inp_sW);
    auto i_ne_offset0 = i_nw_offset0 + iVec(inp_sW);
    auto i_sw_offset0 = i_nw_offset0 + iVec(inp_sH);
    auto i_se_offset0 = i_sw_offset0 + iVec(inp_sW);

    interp_params = compute_interp_params<Vecf, iVec, float>(x1, y1);
    auto nw1 = std::get<4>(interp_params);
    auto ne1 = std::get<5>(interp_params);
    auto sw1 = std::get<6>(interp_params);
    auto se1 = std::get<7>(interp_params);

    auto nw_mask1 = std::get<8>(interp_params);
    auto ne_mask1 = std::get<9>(interp_params);
    auto sw_mask1 = std::get<10>(interp_params);
    auto se_mask1 = std::get<11>(interp_params);

    i_y_n = std::get<12>(interp_params);
    i_x_w = std::get<13>(interp_params);

    auto i_nw_offset1 = i_y_n * iVec(inp_sH) + i_x_w * iVec(inp_sW);
    auto i_ne_offset1 = i_nw_offset1 + iVec(inp_sW);
    auto i_sw_offset1 = i_nw_offset1 + iVec(inp_sH);
    auto i_se_offset1 = i_sw_offset1 + iVec(inp_sW);

    int32_t i_nw_offset[Vec::size()];
    int32_t i_ne_offset[Vec::size()];
    int32_t i_sw_offset[Vec::size()];
    int32_t i_se_offset[Vec::size()];
    i_nw_offset0.store(i_nw_offset);
    i_ne_offset0.store(i_ne_offset);
    i_sw_offset0.store(i_sw_offset);
    i_se_offset0.store(i_se_offset);
    i_nw_offset1.store(i_nw_offset + iVec::size());
    i_ne_offset1.store(i_ne_offset + iVec::size());
    i_sw_offset1.store(i_sw_offset + iVec::size());
    i_se_offset1.store(i_se_offset + iVec::size());

    // Vectorized<float> inp_slice_C0, inp_slice_C1;
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t c = 0; c < C; ++c) {
      auto inp_slice_C_ptr = inp_slice[c].data();
      Vecf nw_mask_copy0 = nw_mask0;
      Vecf ne_mask_copy0 = ne_mask0;
      Vecf sw_mask_copy0 = sw_mask0;
      Vecf se_mask_copy0 = se_mask0;
      Vecf nw_mask_copy1 = nw_mask1;
      Vecf ne_mask_copy1 = ne_mask1;
      Vecf sw_mask_copy1 = sw_mask1;
      Vecf se_mask_copy1 = se_mask1;
      Vecf nw_val0, nw_val1, ne_val0, ne_val1, sw_val0, sw_val1, se_val0, se_val1;
      std::tie(nw_val0, nw_val1) = mask_gather_bfloat16(Vecf(0), Vecf(0), inp_slice_C_ptr, i_nw_offset, nw_mask_copy0, nw_mask_copy1);
      std::tie(ne_val0, ne_val1) = mask_gather_bfloat16(Vecf(0), Vecf(0), inp_slice_C_ptr, i_ne_offset, ne_mask_copy0, ne_mask_copy1);
      std::tie(sw_val0, sw_val1) = mask_gather_bfloat16(Vecf(0), Vecf(0), inp_slice_C_ptr, i_sw_offset, sw_mask_copy0, sw_mask_copy1);
      std::tie(se_val0, se_val1) = mask_gather_bfloat16(Vecf(0), Vecf(0), inp_slice_C_ptr, i_se_offset, se_mask_copy0, se_mask_copy1);

      auto interpolated0 = (nw_val0 * nw0) + (ne_val0 * ne0) + (sw_val0 * sw0) + (se_val0 * se0);
      auto interpolated1 = (nw_val1 * nw1) + (ne_val1 * ne1) + (sw_val1 * sw1) + (se_val1 * se1);
      convert_float_bfloat16(interpolated0, interpolated1).store(out_slice[c].data() + offset, len);
    }
  }

  inline void forward(TensorAccessor<scalar_t, 3>& out_slice,
                      const TensorAccessor<scalar_t, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    forward_bilinear(out_slice, inp_slice, offset, grid_x, grid_y, len);

  }

  template<bool input_requires_grad>
  inline void backward(TensorAccessor<scalar_t, 3>* gInp_slice_ptr,
                       TensorAccessor<scalar_t, 3>& gGrid_slice,
                       const TensorAccessor<scalar_t, 3>& gOut_slice,
                       const TensorAccessor<scalar_t, 3>& inp_slice,
                       int64_t offset, const Vec& grid_x, const Vec& grid_y,
                       int64_t len) const {
    backward_bilinear<input_requires_grad>(gInp_slice_ptr, gGrid_slice, gOut_slice, inp_slice, offset, grid_x, grid_y, len);

  }

  template <bool input_requires_grad, typename scalar_type>
  inline typename std::enable_if_t<!std::is_same<scalar_type, BFloat16>::value>
  backward_bilinear(TensorAccessor<scalar_type, 3>* gInp_slice_ptr,
                    TensorAccessor<scalar_type, 3>& gGrid_slice,
                    const TensorAccessor<scalar_type, 3>& gOut_slice,
                    const TensorAccessor<scalar_type, 3>& inp_slice,
                    int64_t offset, const Vectorized<scalar_type>& grid_x, const Vectorized<scalar_type>& grid_y,
                    int64_t len) const {
    Vec x, y, gx_mult, gy_mult;
    std::tie(x, gx_mult) = compute_W.apply_get_grad(grid_x);
    std::tie(y, gy_mult) = compute_H.apply_get_grad(grid_y);

    Vec n, s, w, e, nw, ne, sw, se, nw_mask, ne_mask, sw_mask, se_mask;
    iVec i_y_n, i_x_w;

    std::tie(
      n, s, w, e, nw, ne, sw, se, nw_mask, ne_mask, sw_mask, se_mask,
      i_y_n, i_x_w) = compute_interp_params<Vec, iVec, scalar_type>(x, y);

    auto i_nw_offset = i_y_n * iVec(inp_sH) + i_x_w * iVec(inp_sW);
    auto i_ne_offset = i_nw_offset + iVec(inp_sW);
    auto i_sw_offset = i_nw_offset + iVec(inp_sH);
    auto i_se_offset = i_sw_offset + iVec(inp_sW);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_nw_mask_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_ne_mask_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_sw_mask_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_se_mask_arr[iVec::size()];
    nw_mask.store(i_nw_mask_arr);
    ne_mask.store(i_ne_mask_arr);
    sw_mask.store(i_sw_mask_arr);
    se_mask.store(i_se_mask_arr);

    // i_gInp_*_offset_arr and gInp_corner_arr variables below are unnecessary
    // when input_requires_grad is false (they are only used within the
    // if-blocks), but required to make the code well-formed.

    // When reading input values, we used mask_gather. Unfortunately, there is
    // no mask_scatter_add (the backward of mask_gather) in Intel intrinsics.
    // So we store the necessary vectors to temporary arrays and use the helper
    // mask_scatter_add defined above.

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_gInp_nw_offset_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_gInp_ne_offset_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_gInp_sw_offset_arr[iVec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_gInp_se_offset_arr[iVec::size()];
    if (input_requires_grad) {
      auto i_gInp_nw_offset = i_y_n * iVec(inp_W) + i_x_w;
      auto i_gInp_ne_offset = i_gInp_nw_offset + iVec(1);
      auto i_gInp_sw_offset = i_gInp_nw_offset + iVec(inp_W);
      auto i_gInp_se_offset = i_gInp_sw_offset + iVec(1);

      i_gInp_nw_offset.store(i_gInp_nw_offset_arr);
      i_gInp_ne_offset.store(i_gInp_ne_offset_arr);
      i_gInp_sw_offset.store(i_gInp_sw_offset_arr);
      i_gInp_se_offset.store(i_gInp_se_offset_arr);
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    scalar_t gInp_corner_arr[Vec::size()];

    auto gx = Vec(0), gy = Vec(0);
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (const auto c : c10::irange(C)) {
      auto inp_slice_C_ptr = inp_slice[c].data();
      auto gOut = Vec::loadu(gOut_slice[c].data() + offset, len);

      if (input_requires_grad) {
        auto gInp_slice_C_ptr = (*gInp_slice_ptr)[c].data();

        (nw * gOut).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_nw_offset_arr, i_nw_mask_arr, len);
        (ne * gOut).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_ne_offset_arr, i_ne_mask_arr, len);
        (sw * gOut).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_sw_offset_arr, i_sw_mask_arr, len);
        (se * gOut).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_se_offset_arr, i_se_mask_arr, len);
      }

      // mask_gather zeros out the mask, so we need to make copies
      Vec nw_mask_copy = nw_mask;
      Vec ne_mask_copy = ne_mask;
      Vec sw_mask_copy = sw_mask;
      Vec se_mask_copy = se_mask;
      auto nw_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_nw_offset, nw_mask_copy);
      auto ne_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_ne_offset, ne_mask_copy);
      auto sw_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_sw_offset, sw_mask_copy);
      auto se_val = mask_gather<sizeof(scalar_t)>(Vec(0), inp_slice_C_ptr, i_se_offset, se_mask_copy);

      gx = gx + ((ne_val - nw_val) * s + (se_val - sw_val) * n) * gOut;
      gy = gy + ((sw_val - nw_val) * e + (se_val - ne_val) * w) * gOut;
    }

    gx = gx * gx_mult;
    gy = gy * gy_mult;

    constexpr int64_t step = Vec::size();
    auto interleaved_gGrid = interleave2(gx, gy);
    auto gGrid_ptr = gGrid_slice.data() + offset * 2;
    std::get<0>(interleaved_gGrid).store(gGrid_ptr,
                                         std::min(len * 2, step));
    std::get<1>(interleaved_gGrid).store(gGrid_ptr + step,
                                         std::max(static_cast<int64_t>(0), len * 2 - step));
  }

  template <bool input_requires_grad>
  inline void backward_bilinear(TensorAccessor<BFloat16, 3>* gInp_slice_ptr,
                       TensorAccessor<BFloat16, 3>& gGrid_slice,
                       const TensorAccessor<BFloat16, 3>& gOut_slice,
                       const TensorAccessor<BFloat16, 3>& inp_slice,
                       int64_t offset, const Vectorized<BFloat16>& grid_x, const Vectorized<BFloat16>& grid_y,
                       int64_t len) const {
    using Vecf = Vectorized<float>;
    Vecf x0, x1, gx_mult0, gx_mult1, y0, y1, gy_mult0, gy_mult1, grid_x0, grid_x1, grid_y0, grid_y1;
    std::tie(grid_x0, grid_x1) = convert_bfloat16_float(grid_x);
    std::tie(grid_y0, grid_y1) = convert_bfloat16_float(grid_y);

    std::tie(x0, gx_mult0) = compute_W.apply_get_grad(grid_x0);
    std::tie(y0, gy_mult0) = compute_H.apply_get_grad(grid_y0);
    std::tie(x1, gx_mult1) = compute_W.apply_get_grad(grid_x1);
    std::tie(y1, gy_mult1) = compute_H.apply_get_grad(grid_y1);

    Vecf n0, n1, s0, s1, w0, w1, e0, e1, nw0, nw1, ne0, ne1, sw0, sw1, se0, se1,
    nw_mask0, nw_mask1, ne_mask0, ne_mask1, sw_mask0, sw_mask1, se_mask0, se_mask1;
    iVec i_y_n, i_x_w;

    std::tie(
      n0, s0, w0, e0, nw0, ne0, sw0, se0, nw_mask0, ne_mask0, sw_mask0, se_mask0,
      i_y_n, i_x_w) = compute_interp_params<Vecf, iVec, float>(x0, y0);

    auto i_nw_offset0 = i_y_n * iVec(inp_sH) + i_x_w * iVec(inp_sW);
    auto i_ne_offset0 = i_nw_offset0 + iVec(inp_sW);
    auto i_sw_offset0 = i_nw_offset0 + iVec(inp_sH);
    auto i_se_offset0 = i_sw_offset0 + iVec(inp_sW);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    int32_t i_nw_mask_arr[Vec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    int32_t i_ne_mask_arr[Vec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    int32_t i_sw_mask_arr[Vec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    int32_t i_se_mask_arr[Vec::size()];
    nw_mask0.store(i_nw_mask_arr);
    ne_mask0.store(i_ne_mask_arr);
    sw_mask0.store(i_sw_mask_arr);
    se_mask0.store(i_se_mask_arr);

    // i_gInp_*_offset_arr and gInp_corner_arr variables below are unnecessary
    // when input_requires_grad is false (they are only used within the
    // if-blocks), but required to make the code well-formed.

    // When reading input values, we used mask_gather. Unfortunately, there is
    // no mask_scatter_add (the backward of mask_gather) in Intel intrinsics.
    // So we store the necessary vectors to temporary arrays and use the helper
    // mask_scatter_add defined above.

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    int32_t i_gInp_nw_offset_arr[Vec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    int32_t i_gInp_ne_offset_arr[Vec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    int32_t i_gInp_sw_offset_arr[Vec::size()];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    int32_t i_gInp_se_offset_arr[Vec::size()];
    if (input_requires_grad) {
      auto i_gInp_nw_offset = i_y_n * iVec(inp_W) + i_x_w;
      auto i_gInp_ne_offset = i_gInp_nw_offset + iVec(1);
      auto i_gInp_sw_offset = i_gInp_nw_offset + iVec(inp_W);
      auto i_gInp_se_offset = i_gInp_sw_offset + iVec(1);

      i_gInp_nw_offset.store(i_gInp_nw_offset_arr);
      i_gInp_ne_offset.store(i_gInp_ne_offset_arr);
      i_gInp_sw_offset.store(i_gInp_sw_offset_arr);
      i_gInp_se_offset.store(i_gInp_se_offset_arr);
    }

    //======================================================
      std::tie(
      n1, s1, w1, e1, nw1, ne1, sw1, se1, nw_mask1, ne_mask1, sw_mask1, se_mask1,
      i_y_n, i_x_w) = compute_interp_params<Vecf, iVec, float>(x1, y1);

    auto i_nw_offset1 = i_y_n * iVec(inp_sH) + i_x_w * iVec(inp_sW);
    auto i_ne_offset1 = i_nw_offset1 + iVec(inp_sW);
    auto i_sw_offset1 = i_nw_offset1 + iVec(inp_sH);
    auto i_se_offset1 = i_sw_offset1 + iVec(inp_sW);

    if (input_requires_grad) {
      auto i_gInp_nw_offset = i_y_n * iVec(inp_W) + i_x_w;
      auto i_gInp_ne_offset = i_gInp_nw_offset + iVec(1);
      auto i_gInp_sw_offset = i_gInp_nw_offset + iVec(inp_W);
      auto i_gInp_se_offset = i_gInp_sw_offset + iVec(1);

      i_gInp_nw_offset.store(i_gInp_nw_offset_arr + iVec::size());
      i_gInp_ne_offset.store(i_gInp_ne_offset_arr + iVec::size());
      i_gInp_sw_offset.store(i_gInp_sw_offset_arr + iVec::size());
      i_gInp_se_offset.store(i_gInp_se_offset_arr + iVec::size());
    }

    nw_mask1.store(i_nw_mask_arr + iVec::size());
    ne_mask1.store(i_ne_mask_arr + iVec::size());
    sw_mask1.store(i_sw_mask_arr + iVec::size());
    se_mask1.store(i_se_mask_arr + iVec::size());

    int32_t i_nw_offset[Vec::size()];
    int32_t i_ne_offset[Vec::size()];
    int32_t i_sw_offset[Vec::size()];
    int32_t i_se_offset[Vec::size()];
    i_nw_offset0.store(i_nw_offset);
    i_ne_offset0.store(i_ne_offset);
    i_sw_offset0.store(i_sw_offset);
    i_se_offset0.store(i_se_offset);
    i_nw_offset1.store(i_nw_offset + iVec::size());
    i_ne_offset1.store(i_ne_offset + iVec::size());
    i_sw_offset1.store(i_sw_offset + iVec::size());
    i_se_offset1.store(i_se_offset + iVec::size());

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    BFloat16 gInp_corner_arr[Vec::size()];

    auto gx0 = Vecf(0), gx1 = Vecf(0), gy0 = Vecf(0), gy1 = Vecf(0);

    Vecf gOut0, gOut1;

    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t c = 0; c < C; ++c) {
      auto inp_slice_C_ptr = inp_slice[c].data();
      std::tie(gOut0, gOut1) = convert_bfloat16_float(Vec::loadu(gOut_slice[c].data() + offset, len));

      if (input_requires_grad) {
        auto gInp_slice_C_ptr = (*gInp_slice_ptr)[c].data();
        convert_float_bfloat16(nw0 * gOut0, nw1 * gOut1).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_nw_offset_arr, i_nw_mask_arr, len);
        convert_float_bfloat16(ne0 * gOut0, ne1 * gOut1).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_ne_offset_arr, i_ne_mask_arr, len);
        convert_float_bfloat16(sw0 * gOut0, sw1 * gOut1).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_sw_offset_arr, i_sw_mask_arr, len);
        convert_float_bfloat16(se0 * gOut0, se1 * gOut1).store(gInp_corner_arr);
        mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_se_offset_arr, i_se_mask_arr, len);
      }
      Vecf nw_mask_copy0 = nw_mask0;
      Vecf ne_mask_copy0 = ne_mask0;
      Vecf sw_mask_copy0 = sw_mask0;
      Vecf se_mask_copy0 = se_mask0;
      Vecf nw_mask_copy1 = nw_mask1;
      Vecf ne_mask_copy1 = ne_mask1;
      Vecf sw_mask_copy1 = sw_mask1;
      Vecf se_mask_copy1 = se_mask1;
      Vecf nw_val0, nw_val1, ne_val0, ne_val1, sw_val0, sw_val1, se_val0, se_val1;
      std::tie(nw_val0, nw_val1) = mask_gather_bfloat16(Vecf(0), Vecf(0), inp_slice_C_ptr, i_nw_offset, nw_mask_copy0, nw_mask_copy1);
      std::tie(ne_val0, ne_val1) = mask_gather_bfloat16(Vecf(0), Vecf(0), inp_slice_C_ptr, i_ne_offset, ne_mask_copy0, ne_mask_copy1);
      std::tie(sw_val0, sw_val1) = mask_gather_bfloat16(Vecf(0), Vecf(0), inp_slice_C_ptr, i_sw_offset, sw_mask_copy0, sw_mask_copy1);
      std::tie(se_val0, se_val1) = mask_gather_bfloat16(Vecf(0), Vecf(0), inp_slice_C_ptr, i_se_offset, se_mask_copy0, se_mask_copy1);

      gx0 = gx0 + ((ne_val0 - nw_val0) * s0 + (se_val0 - sw_val0) * n0) * gOut0;
      gy0 = gy0 + ((sw_val0 - nw_val0) * e0 + (se_val0 - ne_val0) * w0) * gOut0;
      gx1 = gx1 + ((ne_val1 - nw_val1) * s1 + (se_val1 - sw_val1) * n1) * gOut1;
      gy1 = gy1 + ((sw_val1 - nw_val1) * e1 + (se_val1 - ne_val1) * w1) * gOut1;
    }

    gx0 = gx0 * gx_mult0;
    gy0 = gy0 * gy_mult0;
    gx1 = gx1 * gx_mult1;
    gy1 = gy1 * gy_mult1;

    constexpr int64_t step = Vec::size();
    auto interleaved_gGrid = interleave2(convert_float_bfloat16(gx0, gx1), convert_float_bfloat16(gy0, gy1));
    auto gGrid_ptr = gGrid_slice.data() + offset * 2;
    std::get<0>(interleaved_gGrid).store(gGrid_ptr,
                                         std::min(len * 2, step));
    std::get<1>(interleaved_gGrid).store(gGrid_ptr + step,
                                         std::max(static_cast<int64_t>(0), len * 2 - step));
  }
};

template<typename scalar_t, GridSamplerPadding padding, bool align_corners>
struct ApplyGridSample<scalar_t, 2, GridSamplerInterpolation::Nearest,
                       padding, align_corners> {
  using Vec = Vectorized<scalar_t>;
  using integer_t = typename std::conditional_t<std::is_same<scalar_t, BFloat16>::value, int32_t, int_same_size_t<scalar_t>>;
  using iVec = Vectorized<integer_t>;
  using scalar_tb = typename std::conditional_t<std::is_same<scalar_t, BFloat16>::value, float, scalar_t>;

  const int64_t inp_H;
  const int64_t inp_W;
  const int64_t inp_sH;
  const int64_t inp_sW;
  const int64_t C;
  const int64_t inp_sC;
  const ComputeLocation<scalar_tb, padding, align_corners> compute_H;
  const ComputeLocation<scalar_tb, padding, align_corners> compute_W;
  const bool must_in_bound = padding != GridSamplerPadding::Zeros;

  ApplyGridSample(const TensorAccessor<scalar_t, 4>& input)
    : inp_H(input.size(2))
    , inp_W(input.size(3))
    , inp_sH(input.stride(2))
    , inp_sW(input.stride(3))
    , C(input.size(1))
    , inp_sC(input.stride(1))
    , compute_H(input.size(2))
    , compute_W(input.size(3)) {}

  inline void forward(TensorAccessor<scalar_t, 3>& out_slice,
                      const TensorAccessor<scalar_t, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    forward_nearest(out_slice, inp_slice, offset, grid_x, grid_y, len);
  }


  template <typename scalar_type>
  inline typename std::enable_if_t<!std::is_same<scalar_type, BFloat16>::value>
  forward_nearest(TensorAccessor<scalar_type, 3>& out_slice,
                      const TensorAccessor<scalar_type, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    auto x = compute_W.apply(grid_x);
    auto y = compute_H.apply(grid_y);

    auto x_nearest = x.round();
    auto y_nearest = y.round();

    auto i_x_nearest = convert_to_int_of_same_size(x_nearest);
    auto i_y_nearest = convert_to_int_of_same_size(y_nearest);

    auto i_mask = must_in_bound ? iVec(-1)
                                : (i_x_nearest > iVec(-1)) & (i_x_nearest < iVec(inp_W)) &
                                  (i_y_nearest > iVec(-1)) & (i_y_nearest < iVec(inp_H));
    auto mask = cast<scalar_type>(i_mask);

    auto i_offset = i_y_nearest * iVec(inp_sH) + i_x_nearest * iVec(inp_sW);

    auto out_ptr = out_slice.data() + offset;
    auto out_sC = out_slice.stride(0);
    auto inp_slice_ptr = inp_slice.data();
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t c = 0; c < C; ++c, out_ptr += out_sC, inp_slice_ptr += inp_sC) {
      // mask_gather zeros out the mask, so we need to make a copy
      auto mask_copy = mask;
      auto inp_val = mask_gather<sizeof(scalar_type)>(Vec(0), inp_slice_ptr, i_offset, mask_copy);
      inp_val.store(static_cast<void*>(out_ptr), len);
    }
  }

  inline void forward_nearest(TensorAccessor<BFloat16, 3>& out_slice,
                      const TensorAccessor<BFloat16, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    using Vecf = Vectorized<float>;
    Vecf grid_x0, grid_x1, grid_y0, grid_y1;
    std::tie(grid_x0, grid_x1) = convert_bfloat16_float(grid_x);
    std::tie(grid_y0, grid_y1) = convert_bfloat16_float(grid_y);
    auto x = compute_W.apply(grid_x0);
    auto y = compute_H.apply(grid_y0);

    auto x_nearest = x.round();
    auto y_nearest = y.round();

    auto i_x_nearest = convert_to_int_of_same_size(x_nearest);
    auto i_y_nearest = convert_to_int_of_same_size(y_nearest);

    auto i_mask = must_in_bound ? iVec(-1)
                                : (i_x_nearest > iVec(-1)) & (i_x_nearest < iVec(inp_W)) &
                                  (i_y_nearest > iVec(-1)) & (i_y_nearest < iVec(inp_H));
    auto mask0 = cast<float>(i_mask);

    auto i_offset0 = i_y_nearest * iVec(inp_sH) + i_x_nearest * iVec(inp_sW);
    int32_t i_offset[Vec::size()];
    i_offset0.store(i_offset);
    //===================================================================================
    x = compute_W.apply(grid_x1);
    y = compute_H.apply(grid_y1);

    x_nearest = x.round();
    y_nearest = y.round();

    i_x_nearest = convert_to_int_of_same_size(x_nearest);
    i_y_nearest = convert_to_int_of_same_size(y_nearest);

    i_mask = must_in_bound ? iVec(-1)
                                : (i_x_nearest > iVec(-1)) & (i_x_nearest < iVec(inp_W)) &
                                  (i_y_nearest > iVec(-1)) & (i_y_nearest < iVec(inp_H));
    auto mask1 = cast<float>(i_mask);
    auto i_offset1 = i_y_nearest * iVec(inp_sH) + i_x_nearest * iVec(inp_sW);
    i_offset1.store(i_offset + iVec::size());
    auto out_ptr = out_slice.data() + offset;
    auto out_sC = out_slice.stride(0);
    auto inp_slice_ptr = inp_slice.data();
    Vecf inp_val0, inp_val1;
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t c = 0; c < C; ++c, out_ptr += out_sC, inp_slice_ptr += inp_sC) {
      // mask_gather zeros out the mask, so we need to make a copy
      auto mask_copy0 = mask0;
      auto mask_copy1 = mask1;
      std::tie(inp_val0, inp_val1) = mask_gather_bfloat16(Vecf(0), Vecf(0), inp_slice_ptr, i_offset, mask_copy0, mask_copy1);
      convert_float_bfloat16(inp_val0, inp_val1).store(static_cast<void*>(out_ptr), len);
    }
  }

  template<bool input_requires_grad>
  inline void backward(TensorAccessor<scalar_t, 3>* gInp_slice_ptr,
                       TensorAccessor<scalar_t, 3>& gGrid_slice,
                       const TensorAccessor<scalar_t, 3>& gOut_slice,
                       const TensorAccessor<scalar_t, 3>& inp_slice,
                       int64_t offset, const Vec& grid_x, const Vec& grid_y,
                       int64_t len) const {
    backward_nearest<input_requires_grad>(gInp_slice_ptr, gGrid_slice, gOut_slice, inp_slice, offset, grid_x, grid_y, len);
  }

  template <bool input_requires_grad, typename scalar_type>
  inline typename std::enable_if_t<!std::is_same<scalar_type, BFloat16>::value>
  backward_nearest(TensorAccessor<scalar_type, 3>* gInp_slice_ptr,
                       TensorAccessor<scalar_type, 3>& gGrid_slice,
                       const TensorAccessor<scalar_type, 3>& gOut_slice,
                       const TensorAccessor<scalar_type, 3>& inp_slice,
                       int64_t offset, const Vec& grid_x, const Vec& grid_y,
                       int64_t len) const {
    if (input_requires_grad) {
      auto x = compute_W.apply(grid_x);
      auto y = compute_H.apply(grid_y);

      auto x_nearest = x.round();
      auto y_nearest = y.round();

      auto i_x_nearest = convert_to_int_of_same_size(x_nearest);
      auto i_y_nearest = convert_to_int_of_same_size(y_nearest);

      auto i_mask = must_in_bound ? iVec(-1)
                                  : (i_x_nearest > iVec(-1)) & (i_x_nearest < iVec(inp_W)) &
                                    (i_y_nearest > iVec(-1)) & (i_y_nearest < iVec(inp_H));

      auto i_gInp_offset = i_y_nearest * iVec(inp_W) + i_x_nearest;  // gInp is contiguous

      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      integer_t mask_arr[iVec::size()];
      i_mask.store(mask_arr);
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      integer_t gInp_offset_arr[iVec::size()];
      i_gInp_offset.store(gInp_offset_arr);

      #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (const auto c : c10::irange(C)) {
        mask_scatter_add(gOut_slice[c].data() + offset, (*gInp_slice_ptr)[c].data(),
                        gInp_offset_arr, mask_arr, len);
      }
    }
    // grid has zero 0 gradient in Nearest mode
    auto gGrid_ptr = gGrid_slice.data() + offset * 2;
    std::memset(gGrid_ptr, 0, sizeof(scalar_type) * len * 2);
  }

  template <bool input_requires_grad>
  inline void backward_nearest(TensorAccessor<BFloat16, 3>* gInp_slice_ptr,
                       TensorAccessor<BFloat16, 3>& gGrid_slice,
                       const TensorAccessor<BFloat16, 3>& gOut_slice,
                       const TensorAccessor<BFloat16, 3>& inp_slice,
                       int64_t offset, const Vec& grid_x, const Vec& grid_y,
                       int64_t len) const {
    if (input_requires_grad) {
      using Vecf = Vectorized<float>;
      Vecf grid_x0, grid_x1, grid_y0, grid_y1;
      std::tie(grid_x0, grid_x1) = convert_bfloat16_float(grid_x);
      std::tie(grid_y0, grid_y1) = convert_bfloat16_float(grid_y);
      auto x = compute_W.apply(grid_x0);
      auto y = compute_H.apply(grid_y0);

      auto x_nearest = x.round();
      auto y_nearest = y.round();

      auto i_x_nearest = convert_to_int_of_same_size(x_nearest);
      auto i_y_nearest = convert_to_int_of_same_size(y_nearest);

      auto i_mask0 = must_in_bound ? iVec(-1)
                                  : (i_x_nearest > iVec(-1)) & (i_x_nearest < iVec(inp_W)) &
                                    (i_y_nearest > iVec(-1)) & (i_y_nearest < iVec(inp_H));

      auto i_gInp_offset0 = i_y_nearest * iVec(inp_W) + i_x_nearest;  // gInp is contiguous

      //==============================================================================
      x = compute_W.apply(grid_x1);
      y = compute_H.apply(grid_y1);

      x_nearest = x.round();
      y_nearest = y.round();

      i_x_nearest = convert_to_int_of_same_size(x_nearest);
      i_y_nearest = convert_to_int_of_same_size(y_nearest);

      auto i_mask1 = must_in_bound ? iVec(-1)
                                  : (i_x_nearest > iVec(-1)) & (i_x_nearest < iVec(inp_W)) &
                                    (i_y_nearest > iVec(-1)) & (i_y_nearest < iVec(inp_H));

      auto i_gInp_offset1 = i_y_nearest * iVec(inp_W) + i_x_nearest;  // gInp is contiguous

      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      integer_t mask_arr[Vec::size()];
      i_mask0.store(mask_arr);
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      integer_t gInp_offset_arr[Vec::size()];
      i_gInp_offset0.store(gInp_offset_arr);

      i_mask1.store(mask_arr + iVec::size());
      i_gInp_offset1.store(gInp_offset_arr + iVec::size());

      #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (int64_t c = 0; c < C; ++c) {
        mask_scatter_add(gOut_slice[c].data() + offset, (*gInp_slice_ptr)[c].data(),
                        gInp_offset_arr, mask_arr, len);
      }
    }

    // grid has zero 0 gradient in Nearest mode
    auto gGrid_ptr = gGrid_slice.data() + offset * 2;
    std::memset(gGrid_ptr, 0, sizeof(BFloat16) * len * 2);
  }
};

// Use bicubic convolution algorithm. Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template<typename scalar_t, GridSamplerPadding padding, bool align_corners>
struct ApplyGridSample<scalar_t, 2, GridSamplerInterpolation::Bicubic,
                       padding, align_corners> {
  using Vec = Vectorized<scalar_t>;
  using integer_t = typename std::conditional_t<std::is_same<scalar_t, BFloat16>::value, int32_t, int_same_size_t<scalar_t>>;
  using iVec = Vectorized<integer_t>;
  using scalar_tb = typename std::conditional_t<std::is_same<scalar_t, BFloat16>::value, float, scalar_t>;

  const int64_t inp_H;
  const int64_t inp_W;
  const int64_t inp_sH;
  const int64_t inp_sW;
  const int64_t C;
  const int64_t inp_sC;
  const ComputeLocation<scalar_tb, padding, align_corners> compute_H;
  const ComputeLocation<scalar_tb, padding, align_corners> compute_W;
  const bool must_in_bound = padding != GridSamplerPadding::Zeros;

  // constant used in cubic convolution
  // could be -0.5 or -0.75, use the same value in UpSampleBicubic2d.h
  Vectorized<scalar_tb> A = Vectorized<scalar_tb>(-0.75);

  ApplyGridSample(const TensorAccessor<scalar_t, 4>& input)
    : inp_H(input.size(2))
    , inp_W(input.size(3))
    , inp_sH(input.stride(2))
    , inp_sW(input.stride(3))
    , C(input.size(1))
    , inp_sC(input.stride(1))
    , compute_H(input.size(2))
    , compute_W(input.size(3)) {}

  // Calculate the cubic convolution coefficient
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  template<typename Vec_t>
  inline void get_cubic_coefficients(Vec_t (&coeffs)[4], const Vec_t& tx) const {
    Vec_t x;
    x = tx + Vec_t(1);  // 1 < x = |-1 - tx| < 2
    coeffs[0] = ((A * x - Vec_t(5) * A) * x + Vec_t(8) * A) * x - Vec_t(4) * A;
    x = tx;           // x = |0 - tx| <= 1
    coeffs[1] = ((A + Vec_t(2)) * x - (A + Vec_t(3))) * x * x + Vec_t(1);
    x = Vec_t(1) - tx;  // x = |1 - tx| <= 1
    coeffs[2] = ((A + Vec_t(2)) * x - (A + Vec_t(3))) * x * x + Vec_t(1);
    x = Vec_t(2) - tx;  // 1 < x = |2 - tx| < 2
    coeffs[3] = ((A * x - Vec_t(5) * A) * x + Vec_t(8) * A) * x - Vec_t(4) * A;
  }

  // Calculate the differential of the cubic convolution, i.e. `d coeff / d x`
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  template<typename Vec_t>
  inline void get_cubic_coefficients_grad(Vec_t (&coeffs)[4], const Vec_t& tx) const {
    Vec_t x;
    x = Vec_t(-1) - tx; // 1 < x = |-1 - tx| < 2
    coeffs[0] = (Vec_t(-3) * A * x - Vec_t(10) * A ) * x - Vec_t(8) * A;
    x = Vec_t(0) - tx;  // x = |0 - tx| <= 1
    coeffs[1] = (Vec_t(-3) * (A + Vec_t(2)) * x - Vec_t(2) * (A + Vec_t(3))) * x;
    x = Vec_t(1) - tx;  // x = |1 - tx| <= 1
    coeffs[2] = (Vec_t(3) * (A + Vec_t(2)) * x - Vec_t(2) * (A + Vec_t(3))) * x;
    x = Vec_t(2) - tx;  // 1 < x = |2 - tx| < 2
    coeffs[3] = (Vec_t(3) * A * x - Vec_t(10) * A) * x + Vec_t(8) * A;
  }

  template<typename scalar_type>
  typename std::enable_if_t<!std::is_same<scalar_type, BFloat16>::value, Vectorized<scalar_type>>
  inline get_value_bounded(const scalar_type* data, const Vectorized<scalar_type>& x,
                              const Vectorized<scalar_type>& y) const {
    auto ix = convert_to_int_of_same_size(compute_W.compute_coordinates(x));
    auto iy = convert_to_int_of_same_size(compute_H.compute_coordinates(y));

    auto mask_x = must_in_bound ? iVec(-1) : (ix > iVec(-1)) & (ix < iVec(inp_W));
    auto mask_y = must_in_bound ? iVec(-1) : (iy > iVec(-1)) & (iy < iVec(inp_H));
    auto mask = cast<scalar_type>(mask_x & mask_y);

    auto offset = iy * iVec(inp_sH) + ix * iVec(inp_sW);

    auto val = mask_gather<sizeof(scalar_type)>(Vectorized<scalar_type>(0), data, offset, mask);
    return val;
  }

  std::tuple<Vectorized<float>, Vectorized<float>>
  inline get_value_bounded_bfloat16(const BFloat16* data, const Vectorized<float>& x0, const Vectorized<float>& x1,
                              const Vectorized<float>& y0, const Vectorized<float>& y1) const {
    auto ix = convert_to_int_of_same_size(compute_W.compute_coordinates(x0));
    auto iy = convert_to_int_of_same_size(compute_H.compute_coordinates(y0));

    auto mask_x = must_in_bound ? iVec(-1) : (ix > iVec(-1)) & (ix < iVec(inp_W));
    auto mask_y = must_in_bound ? iVec(-1) : (iy > iVec(-1)) & (iy < iVec(inp_H));
    auto mask0 = cast<float>(mask_x & mask_y);

    auto offset0 = iy * iVec(inp_sH) + ix * iVec(inp_sW);
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ix = convert_to_int_of_same_size(compute_W.compute_coordinates(x1));
    iy = convert_to_int_of_same_size(compute_H.compute_coordinates(y1));

    mask_x = must_in_bound ? iVec(-1) : (ix > iVec(-1)) & (ix < iVec(inp_W));
    mask_y = must_in_bound ? iVec(-1) : (iy > iVec(-1)) & (iy < iVec(inp_H));
    auto mask1 = cast<float>(mask_x & mask_y);

    auto offset1 = iy * iVec(inp_sH) + ix * iVec(inp_sW);
    int32_t offset[Vec::size()];
    offset0.store(offset);
    offset1.store(offset + iVec::size());

    auto val = mask_gather_bfloat16(Vectorized<float>(0), Vectorized<float>(0), data, offset, mask0, mask1);
    return val;
  }

  template<typename scalar_type>
  inline typename std::enable_if_t<!std::is_same<scalar_type, BFloat16>::value>
  add_value_bounded(scalar_type* data, int64_t len, const Vectorized<scalar_type>& x,
                              const Vectorized<scalar_type>&y,const Vectorized<scalar_type>& delta) const {
    auto ix = convert_to_int_of_same_size(compute_W.compute_coordinates(x));
    auto iy = convert_to_int_of_same_size(compute_H.compute_coordinates(y));

    auto mask_x = must_in_bound ? iVec(-1) : (ix > iVec(-1)) & (ix < iVec(inp_W));
    auto mask_y = must_in_bound ? iVec(-1) : (iy > iVec(-1)) & (iy < iVec(inp_H));
    auto mask = cast<scalar_type>(mask_x & mask_y);

    auto i_gInp_offset = iy * iVec(inp_W) + ix;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_gInp_offset_arr[iVec::size()];
    i_gInp_offset.store(i_gInp_offset_arr);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t mask_arr[iVec::size()];
    mask.store(mask_arr);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    scalar_type gInp_corner_arr[Vec::size()];
    delta.store(gInp_corner_arr);

    mask_scatter_add(gInp_corner_arr, data, i_gInp_offset_arr, mask_arr, len);
  }

  inline void add_value_bounded(BFloat16* data, int64_t len, const Vectorized<float>& x0, const Vectorized<float>& x1,
                              const Vectorized<float>&y0, const Vectorized<float>&y1, const Vectorized<BFloat16>& delta) const {
    auto ix0 = convert_to_int_of_same_size(compute_W.compute_coordinates(x0));
    auto iy0 = convert_to_int_of_same_size(compute_H.compute_coordinates(y0));

    auto mask_x = must_in_bound ? iVec(-1) : (ix0 > iVec(-1)) & (ix0 < iVec(inp_W));
    auto mask_y = must_in_bound ? iVec(-1) : (iy0 > iVec(-1)) & (iy0 < iVec(inp_H));
    auto mask0 = cast<float>(mask_x & mask_y);

    auto ix1 = convert_to_int_of_same_size(compute_W.compute_coordinates(x1));
    auto iy1 = convert_to_int_of_same_size(compute_H.compute_coordinates(y1));

    mask_x = must_in_bound ? iVec(-1) : (ix1 > iVec(-1)) & (ix1 < iVec(inp_W));
    mask_y = must_in_bound ? iVec(-1) : (iy1 > iVec(-1)) & (iy1 < iVec(inp_H));
    auto mask1 = cast<float>(mask_x & mask_y);

    auto i_gInp_offset0 = iy0 * iVec(inp_W) + ix0;
    auto i_gInp_offset1 = iy1 * iVec(inp_W) + ix1;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t i_gInp_offset_arr[Vec::size()];
    i_gInp_offset0.store(i_gInp_offset_arr);
    i_gInp_offset1.store(i_gInp_offset_arr + iVec::size());

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    integer_t mask_arr[Vec::size()];
    mask0.store(mask_arr);
    mask1.store(mask_arr + iVec::size());

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    BFloat16 gInp_corner_arr[Vec::size()];
    delta.store(gInp_corner_arr);

    mask_scatter_add(gInp_corner_arr, data, i_gInp_offset_arr, mask_arr, len);
  }

  inline void forward(TensorAccessor<scalar_t, 3>& out_slice,
                      const TensorAccessor<scalar_t, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    forward_bicubic(out_slice, inp_slice, offset, grid_x, grid_y, len);
  }

  template <typename scalar_type>
  inline typename std::enable_if_t<!std::is_same<scalar_type, BFloat16>::value>
  forward_bicubic(TensorAccessor<scalar_type, 3>& out_slice,
                      const TensorAccessor<scalar_type, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    auto x = compute_W.unnormalize(grid_x);
    auto y = compute_H.unnormalize(grid_y);

    auto ix = x.floor();
    auto iy = y.floor();

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vec coeff_x[4];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vec coeff_y[4];
    get_cubic_coefficients(coeff_x, x - ix);
    get_cubic_coefficients(coeff_y, y - iy);

    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (const auto c : c10::irange(C)) {
      auto inp_slice_C_ptr = inp_slice[c].data();

      // Interpolate the 4 values in the x direction
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      Vec interp_x[4];
      for (const auto i : c10::irange(4)) {
        interp_x[i] =
          coeff_x[0] * get_value_bounded<scalar_type>(inp_slice_C_ptr, ix - Vec(1), iy + Vec(-1 + i)) +
          coeff_x[1] * get_value_bounded<scalar_type>(inp_slice_C_ptr, ix + Vec(0), iy + Vec(-1 + i)) +
          coeff_x[2] * get_value_bounded<scalar_type>(inp_slice_C_ptr, ix + Vec(1), iy + Vec(-1 + i)) +
          coeff_x[3] * get_value_bounded<scalar_type>(inp_slice_C_ptr, ix + Vec(2), iy + Vec(-1 + i));
      }

      // Interpolate the 4 values in the y direction
      auto interpolated = coeff_y[0] * interp_x[0] + coeff_y[1] * interp_x[1] +
                          coeff_y[2] * interp_x[2] + coeff_y[3] * interp_x[3];
      interpolated.store(out_slice[c].data() + offset, len);
    }
  }

  inline void forward_bicubic(TensorAccessor<BFloat16, 3>& out_slice,
                      const TensorAccessor<BFloat16, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    using Vecf = Vectorized<float>;
    Vecf grid_x0, grid_x1, grid_y0, grid_y1;
    std::tie(grid_x0, grid_x1) = convert_bfloat16_float(grid_x);
    std::tie(grid_y0, grid_y1) = convert_bfloat16_float(grid_y);
    auto x = compute_W.unnormalize(grid_x0);
    auto y = compute_H.unnormalize(grid_y0);

    auto ix0 = x.floor();
    auto iy0 = y.floor();

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_x0[4];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_y0[4];
    get_cubic_coefficients(coeff_x0, x - ix0);
    get_cubic_coefficients(coeff_y0, y - iy0);

    x = compute_W.unnormalize(grid_x1);
    y = compute_H.unnormalize(grid_y1);

    auto ix1 = x.floor();
    auto iy1 = y.floor();

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_x1[4];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_y1[4];
    get_cubic_coefficients(coeff_x1, x - ix1);
    get_cubic_coefficients(coeff_y1, y - iy1);

    Vecf val0, val1, val2, val3, val4, val5, val6, val7;
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t c = 0; c < C; ++c) {
      auto inp_slice_C_ptr = inp_slice[c].data();

      // Interpolate the 4 values in the x direction
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      Vecf interp_x0[4];
      Vecf interp_x1[4];
      for (int64_t i = 0; i < 4; ++i) {
        std::tie(val0, val1) = get_value_bounded_bfloat16(inp_slice_C_ptr, ix0 - Vecf(1), ix1 - Vecf(1), iy0 + Vecf(-1 + i), iy1 + Vecf(-1 + i));
        std::tie(val2, val3) = get_value_bounded_bfloat16(inp_slice_C_ptr, ix0 + Vecf(0), ix1 + Vecf(0), iy0 + Vecf(-1 + i), iy1 + Vecf(-1 + i));
        std::tie(val4, val5) = get_value_bounded_bfloat16(inp_slice_C_ptr, ix0 + Vecf(1), ix1 + Vecf(1), iy0 + Vecf(-1 + i), iy1 + Vecf(-1 + i));
        std::tie(val6, val7) = get_value_bounded_bfloat16(inp_slice_C_ptr, ix0 + Vecf(2), ix1 + Vecf(2), iy0 + Vecf(-1 + i), iy1 + Vecf(-1 + i));
        interp_x0[i] = coeff_x0[0] * val0 + coeff_x0[1] * val2 + coeff_x0[2] * val4 + coeff_x0[3] * val6;
        interp_x1[i] = coeff_x1[0] * val1 + coeff_x1[1] * val3 + coeff_x1[2] * val5 + coeff_x1[3] * val7;
      }

      // Interpolate the 4 values in the y direction
      auto interpolated0 = coeff_y0[0] * interp_x0[0] + coeff_y0[1] * interp_x0[1] +
                          coeff_y0[2] * interp_x0[2] + coeff_y0[3] * interp_x0[3];
      auto interpolated1 = coeff_y1[0] * interp_x1[0] + coeff_y1[1] * interp_x1[1] +
                          coeff_y1[2] * interp_x1[2] + coeff_y1[3] * interp_x1[3];
      convert_float_bfloat16(interpolated0, interpolated1).store(out_slice[c].data() + offset, len);
    }
  }

  template<bool input_requires_grad>
  inline void backward(TensorAccessor<scalar_t, 3>* gInp_slice_ptr,
                      TensorAccessor<scalar_t, 3>& gGrid_slice,
                      const TensorAccessor<scalar_t, 3>& gOut_slice,
                      const TensorAccessor<scalar_t, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    backward_bicubic<input_requires_grad>(gInp_slice_ptr, gGrid_slice, gOut_slice, inp_slice, offset, grid_x, grid_y, len);
  }


  template <bool input_requires_grad, typename scalar_type>
  inline typename std::enable_if_t<!std::is_same<scalar_type, BFloat16>::value>
  backward_bicubic(TensorAccessor<scalar_type, 3>* gInp_slice_ptr,
                      TensorAccessor<scalar_type, 3>& gGrid_slice,
                      const TensorAccessor<scalar_type, 3>& gOut_slice,
                      const TensorAccessor<scalar_type, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    Vec x = compute_W.unnormalize(grid_x);
    Vec y = compute_H.unnormalize(grid_y);
    Vec gx_mult = Vec(compute_W.scaling_factor);
    Vec gy_mult = Vec(compute_H.scaling_factor);

    auto ix = x.floor();
    auto iy = y.floor();

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vec coeff_x[4];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vec coeff_y[4];
    get_cubic_coefficients(coeff_x, x - ix);
    get_cubic_coefficients(coeff_y, y - iy);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vec coeff_x_grad[4];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vec coeff_y_grad[4];
    get_cubic_coefficients_grad(coeff_x_grad, x - ix);
    get_cubic_coefficients_grad(coeff_y_grad, y - iy);

    auto gx = Vec(0), gy = Vec(0);
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (const auto c : c10::irange(C)) {
      auto inp_slice_C_ptr = inp_slice[c].data();
      auto gOut = Vec::loadu(gOut_slice[c].data() + offset, len);

      for (const auto i : c10::irange(4)) {
        for (const auto j : c10::irange(4)) {
          auto xx = ix + Vec(-1 + i);
          auto yy = iy + Vec(-1 + j);

          if (input_requires_grad) {
            auto gInp_slice_C_ptr = (*gInp_slice_ptr)[c].data();
            add_value_bounded(gInp_slice_C_ptr, len, xx, yy, gOut * coeff_x[i] * coeff_y[j]);
          }

          auto val = get_value_bounded(inp_slice_C_ptr, xx, yy);
          gx = gx - val * gOut * coeff_x_grad[i] * coeff_y[j];
          gy = gy - val * gOut * coeff_y_grad[j] * coeff_x[i];
        }
      }
    }

    gx = gx * gx_mult;
    gy = gy * gy_mult;

    constexpr int64_t step = Vec::size();
    auto interleaved_gGrid = interleave2(gx, gy);
    auto gGrid_ptr = gGrid_slice.data() + offset * 2;
    std::get<0>(interleaved_gGrid).store(gGrid_ptr,
                                         std::min(len * 2, step));
    std::get<1>(interleaved_gGrid).store(gGrid_ptr + step,
                                         std::max(static_cast<int64_t>(0), len * 2 - step));
  }

  template <bool input_requires_grad>
  inline void backward_bicubic(TensorAccessor<BFloat16, 3>* gInp_slice_ptr,
                      TensorAccessor<BFloat16, 3>& gGrid_slice,
                      const TensorAccessor<BFloat16, 3>& gOut_slice,
                      const TensorAccessor<BFloat16, 3>& inp_slice,
                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
                      int64_t len) const {
    using Vecf = Vectorized<float>;
    Vecf grid_x0, grid_x1, grid_y0, grid_y1;
    std::tie(grid_x0, grid_x1) = convert_bfloat16_float(grid_x);
    std::tie(grid_y0, grid_y1) = convert_bfloat16_float(grid_y);

    auto x = compute_W.unnormalize(grid_x0);
    auto y = compute_H.unnormalize(grid_y0);
    auto gx_mult0 = Vecf(compute_W.scaling_factor);
    auto gy_mult0 = Vecf(compute_H.scaling_factor);

    auto ix0 = x.floor();
    auto iy0 = y.floor();

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_x0[4];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_y0[4];
    get_cubic_coefficients(coeff_x0, x - ix0);
    get_cubic_coefficients(coeff_y0, y - iy0);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_x_grad0[4];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_y_grad0[4];
    get_cubic_coefficients_grad(coeff_x_grad0, x - ix0);
    get_cubic_coefficients_grad(coeff_y_grad0, y - iy0);

    //=====================================================================
    x = compute_W.unnormalize(grid_x1);
    y = compute_H.unnormalize(grid_y1);
    auto gx_mult1 = Vecf(compute_W.scaling_factor);
    auto gy_mult1 = Vecf(compute_H.scaling_factor);

    auto ix1 = x.floor();
    auto iy1 = y.floor();

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_x1[4];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_y1[4];
    get_cubic_coefficients(coeff_x1, x - ix1);
    get_cubic_coefficients(coeff_y1, y - iy1);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_x_grad1[4];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    Vecf coeff_y_grad1[4];
    get_cubic_coefficients_grad(coeff_x_grad1, x - ix1);
    get_cubic_coefficients_grad(coeff_y_grad1, y - iy1);

    auto gx0 = Vecf(0), gx1 = Vecf(0), gy0 = Vecf(0), gy1 = Vecf(0);
    Vecf inp_slice_C0, inp_slice_C1, gOut0, gOut1, val0, val1;
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t c = 0; c < C; ++c) {
      auto inp_slice_C_ptr = inp_slice[c].data();
      // std::tie(inp_slice_C0, inp_slice_C1) = convert_bfloat16_float(Vec::loadu(inp_slice[c].data()));
      // auto gOut = Vec::loadu(gOut_slice[c].data() + offset, len);
      std::tie(gOut0, gOut1) = convert_bfloat16_float(Vec::loadu(gOut_slice[c].data() + offset, len));

      for (int64_t i = 0; i < 4; ++i) {
        for (int64_t j = 0; j < 4; ++j) {
          auto xx0 = ix0 + Vecf(-1 + i);
          auto yy0 = iy0 + Vecf(-1 + j);
          auto xx1 = ix1 + Vecf(-1 + i);
          auto yy1 = iy1 + Vecf(-1 + j);

          if (input_requires_grad) {
            auto gInp_slice_C_ptr = (*gInp_slice_ptr)[c].data();
            auto delta = convert_float_bfloat16(gOut0 * coeff_x0[i] * coeff_y0[j], gOut1 * coeff_x1[i] * coeff_y1[j]);
            add_value_bounded(gInp_slice_C_ptr, len, xx0, xx1, yy0, yy1, delta);
          }

          std::tie(val0, val1) = get_value_bounded_bfloat16(inp_slice_C_ptr, xx0, xx1, yy0, yy1);

          gx0 = gx0 - val0 * gOut0 * coeff_x_grad0[i] * coeff_y0[j];
          gy0 = gy0 - val0 * gOut0 * coeff_y_grad0[j] * coeff_x0[i];
          gx1 = gx1 - val1 * gOut1 * coeff_x_grad1[i] * coeff_y1[j];
          gy1 = gy1 - val1 * gOut1 * coeff_y_grad1[j] * coeff_x1[i];
        }
      }
    }

    gx0 = gx0 * gx_mult0;
    gy0 = gy0 * gy_mult0;
    gx1 = gx1 * gx_mult1;
    gy1 = gy1 * gy_mult1;

    constexpr int64_t step = Vec::size();
    auto interleaved_gGrid = interleave2(convert_float_bfloat16(gx0, gx1), convert_float_bfloat16(gy0, gy1));
    auto gGrid_ptr = gGrid_slice.data() + offset * 2;
    std::get<0>(interleaved_gGrid).store(gGrid_ptr,
                                         std::min(len * 2, step));
    std::get<1>(interleaved_gGrid).store(gGrid_ptr + step,
                                         std::max(static_cast<int64_t>(0), len * 2 - step));
  }
};

inline Vectorized<BFloat16> gather_bfloat16(BFloat16 const* base_addr, int32_t const * vindex) {
  static constexpr int size = Vectorized<BFloat16>::size();
  BFloat16 buffer[size];
  for (int64_t i = 0; i < size; i++) {
    buffer[i] = base_addr[vindex[i]];
  }
  return Vectorized<BFloat16>::loadu(static_cast<void*>(buffer));
}

// ~~~~~~~~~~~~~~~~~~ grid_sample_2d_grid_slice_iterator ~~~~~~~~~~~~~~~~~~~~~~
// Function to apply a vectorized function on a grid slice tensor (without batch
// dimension).
// See NOTE [ Grid Sample CPU Kernels ] for details.

// forloop_h do a for-loop over H
template<typename scalar_t, typename ApplyFn>
inline void forloop_h(const TensorAccessor<scalar_t, 3>& grid_slice,
                      int64_t step, int64_t grid_sH, int64_t grid_sW,
                      int64_t grid_sCoor, int64_t out_H, int64_t out_W,
                      const ApplyFn &apply_fn){
  using iVec = Vectorized<int_same_size_t<scalar_t>>;
  auto grid_ptr = grid_slice.data();
  int64_t spatial_offset = 0;
  const int64_t i_offset_delta = grid_sW * step;

  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (int64_t h = 0; h < out_H; h++) {
    auto grid_ptr_x = grid_ptr + h * grid_sH;
    auto grid_ptr_y = grid_ptr_x + grid_sCoor;
    auto i_offsets = iVec::arange(0, grid_sW);
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t w = 0; w < out_W; w += step) {
      auto len = std::min(step, out_W - w);
      if (len < step) {
        // prevents illegal memory access, sets the exceeding offsets to zero
        i_offsets = iVec::set(iVec(0), i_offsets, len);
      }

      apply_fn(vec::gather<sizeof(scalar_t)>(grid_ptr_x, i_offsets),
                vec::gather<sizeof(scalar_t)>(grid_ptr_y, i_offsets),
                spatial_offset, len);

      grid_ptr_x += i_offset_delta;
      grid_ptr_y += i_offset_delta;
      spatial_offset += len;
    }
  }
}

// forloop_h do a for-loop over H
template<typename ApplyFn>
inline void forloop_h(const TensorAccessor<BFloat16, 3>& grid_slice,
                      int64_t step, int64_t grid_sH, int64_t grid_sW,
                      int64_t grid_sCoor, int64_t out_H, int64_t out_W,
                       const ApplyFn &apply_fn){
  using iVec = Vectorized<int32_t>;
  auto grid_ptr = grid_slice.data();
  int64_t spatial_offset = 0;
  const int64_t i_offset_delta = grid_sW * step;

  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (int64_t h = 0; h < out_H; h++) {
    auto grid_ptr_x = grid_ptr + h * grid_sH;
    auto grid_ptr_y = grid_ptr_x + grid_sCoor;
    auto i_offsets0 = iVec::arange(0, grid_sW);
    auto i_offsets1 = iVec::arange(grid_sW * iVec::size(), grid_sW);
    int32_t i_offsets[iVec::size()*2];
    i_offsets0.store(i_offsets);
    i_offsets1.store(i_offsets + iVec::size());
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t w = 0; w < out_W; w += step) {
      auto len = std::min(step, out_W - w);
      if (len < step) {
        // prevents illegal memory access, sets the exceeding offsets to zero
        i_offsets0 = iVec::set(iVec(0), i_offsets0, len);
        i_offsets1 = iVec::set(iVec(0), i_offsets1, (len - iVec::size()) < 0 ? 0 : (len - iVec::size()));
        i_offsets0.store(i_offsets);
        i_offsets1.store(i_offsets + iVec::size());
      }

      apply_fn(gather_bfloat16(grid_ptr_x, i_offsets),
                gather_bfloat16(grid_ptr_y, i_offsets),
                spatial_offset, len);

      grid_ptr_x += i_offset_delta;
      grid_ptr_y += i_offset_delta;
      spatial_offset += len;
    }
  }
}

template<typename scalar_t, typename ApplyFn>
static inline void grid_sample_2d_grid_slice_iterator(
    const TensorAccessor<scalar_t, 3>& grid_slice, const ApplyFn &apply_fn) {
  int64_t out_H = grid_slice.size(0);
  int64_t out_W = grid_slice.size(1);
  int64_t grid_sH = grid_slice.stride(0);
  int64_t grid_sW = grid_slice.stride(1);
  int64_t grid_sCoor = grid_slice.stride(2);
  auto grid_ptr = grid_slice.data();

  using Vec = Vectorized<scalar_t>;
  constexpr int64_t step = Vec::size();
  // Loop over each output pixel in grid.
  // We consider the following three cases (after slicing out the batch
  // dimension).
  // See detailed discussions under each if-case.

  if (at::geometry_is_contiguous({out_H, out_W, 2}, {grid_sH, grid_sW, grid_sCoor})) {
    // Case 1:
    // Grid is contiguous.
    // Strategy: Sequentially load two vectors at the same time, and get,
    //           e.g.,  {x0, y0, x1, y1}, {x2, y2, x3, y3}. Then we use
    //           at::vec::deinterleave2 to get x and y vectors.
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
        x = Vec::set(Vec(0), x, len);
        y = Vec::set(Vec(0), y, len);
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
    auto line_fn = [&](const scalar_t *grid_ptr_x, const scalar_t *grid_ptr_y,
                       int64_t out_base_offset, int64_t total_size) {
      for (int64_t i = 0; i < total_size; i += step) {
        auto len = std::min(step, total_size - i);
        auto x = Vec::loadu(grid_ptr_x + i, len);
        auto y = Vec::loadu(grid_ptr_y + i, len);
        // make sure that x and y are valid grid sample locations
        if (len < step) {
          x = Vec::set(Vec(0), x, len);
          y = Vec::set(Vec(0), y, len);
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
      for (const auto h : c10::irange(out_H)) {
        line_fn(grid_ptr_NH, grid_ptr_NH + grid_sCoor, h * out_W, out_W);
        grid_ptr_NH += grid_sH;
      }
    }
  } else {
    // Case 3:
    // General case.
    // Strategy: Do a for-loop over H, for each W slice, use
    //           at::vec::gather to load the x and y vectors.
    forloop_h(grid_slice, step, grid_sH, grid_sW, grid_sCoor, out_H, out_W, apply_fn);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~ Grid Sample Kernels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Use the structs & functions defined above to calculate grid sample forward
// and backward.
// See NOTE [ Grid Sample CPU Kernels ] for details.

Tensor grid_sampler_2d_cpu_kernel_impl(const Tensor& input, const Tensor& grid,
                                       int64_t interpolation_mode,
                                       int64_t padding_mode, bool align_corners) {
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto output = at::empty({N, input.size(1), H, W}, input.options());
  auto spatial_size = H * W;
  auto grain_size = spatial_size == 0 ? (N + 1)
                                      : at::divup(at::internal::GRAIN_SIZE, spatial_size * 4 /* 2d * 2 tensors*/);

#define HANDLE_CASE(interp, padding, align_corners)                            \
  case padding: {                                                              \
    ApplyGridSample<scalar_t, 2, interp, padding, align_corners>               \
    grid_sample(inp_acc);                                                      \
    parallel_for(0, N, grain_size, [&](int64_t begin, int64_t end) {           \
      for (const auto n : c10::irange(begin, end)) {                                  \
        auto out_slice = out_acc[n];                                           \
        auto inp_slice = inp_acc[n];                                           \
        grid_sample_2d_grid_slice_iterator(                                    \
          grid_acc[n],                                                         \
          [&](const Vectorized<scalar_t>& grid_x, const Vectorized<scalar_t>& grid_y,  \
              int64_t spatial_offset, int64_t len) {                           \
            grid_sample.forward(out_slice, inp_slice, spatial_offset,          \
                                grid_x, grid_y, len);                          \
          });                                                                  \
        }                                                                      \
      });                                                                      \
    return;                                                                    \
  }

#define HANDLE_INTERP(interp, align_corners)                                   \
  case interp: {                                                               \
    switch (static_cast<GridSamplerPadding>(padding_mode)) {                   \
      HANDLE_CASE(interp, GridSamplerPadding::Zeros, align_corners);           \
      HANDLE_CASE(interp, GridSamplerPadding::Border, align_corners);          \
      HANDLE_CASE(interp, GridSamplerPadding::Reflection, align_corners);      \
    }                                                                          \
    return;                                                                    \
  }

  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, input.scalar_type(), "grid_sampler_2d_cpu_kernel_impl", [&] {
    auto out_acc = output.accessor<scalar_t, 4>();
    auto inp_acc = input.accessor<scalar_t, 4>();
    auto grid_acc = grid.accessor<scalar_t, 4>();
    if (align_corners) {
      switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
        HANDLE_INTERP(GridSamplerInterpolation::Bilinear, true);
        HANDLE_INTERP(GridSamplerInterpolation::Nearest, true);
        HANDLE_INTERP(GridSamplerInterpolation::Bicubic, true);
      }
    } else {
      switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
        HANDLE_INTERP(GridSamplerInterpolation::Bilinear, false);
        HANDLE_INTERP(GridSamplerInterpolation::Nearest, false);
        HANDLE_INTERP(GridSamplerInterpolation::Bicubic, false);
      }
    }
  });
#undef HANDLE_CASE
#undef HANDLE_INTERP

  return output;
}

std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cpu_kernel_impl(const Tensor& grad_output_,
                                         const Tensor& input,
                                         const Tensor& grid,
                                         int64_t interpolation_mode,
                                         int64_t padding_mode,
                                         bool align_corners,
                                         std::array<bool,2> output_mask) {
  // grad_output should be contiguous most of time. Ensuring that it is
  // contiguous can greatly simplify this code.
  auto grad_output = grad_output_.contiguous();

  // If `input` gradient is not required, we skip computing it -- not needing to create
  // the tensor to hold the gradient can markedly increase performance. (`grid` gradient
  // is always computed.)
  auto input_requires_grad = output_mask[0];

  Tensor grad_input;
  if (input_requires_grad) {
    grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto N = input.size(0);
  auto spatial_size = grid.size(1) * grid.size(2);
  auto grain_size = spatial_size == 0 ? (N + 1)
                                      : at::divup(at::internal::GRAIN_SIZE, spatial_size * 10 /* 2d * 5 tensors*/);

#define GINP_SLICE_PTR_true auto gInp_slice = gInp_acc[n]; auto gInp_slice_ptr = &gInp_slice;
#define GINP_SLICE_PTR_false TensorAccessor<scalar_t, 3>* gInp_slice_ptr = nullptr;
#define GINP_SLICE_PTR(input_requires_grad) GINP_SLICE_PTR_##input_requires_grad

#define HANDLE_CASE(interp, padding, align_corners, input_requires_grad)                              \
  case padding: {                                                                \
    ApplyGridSample<scalar_t, 2, interp, padding, align_corners>                 \
    grid_sample(inp_acc);                                                        \
    parallel_for(0, N, grain_size, [&](int64_t begin, int64_t end) {             \
      for (const auto n : c10::irange(begin, end)) {                             \
        GINP_SLICE_PTR(input_requires_grad)                                      \
        auto gGrid_slice = gGrid_acc[n];                                         \
        auto gOut_slice = gOut_acc[n];                                           \
        auto inp_slice = inp_acc[n];                                             \
        grid_sample_2d_grid_slice_iterator(                                      \
          grid_acc[n],                                                           \
          [&](const Vectorized<scalar_t>& grid_x, const Vectorized<scalar_t>& grid_y,    \
              int64_t spatial_offset, int64_t len) {                             \
            grid_sample.backward<input_requires_grad>(gInp_slice_ptr, gGrid_slice,       \
                                                      gOut_slice, inp_slice,     \
                                                      spatial_offset, grid_x,    \
                                                      grid_y, len);              \
          });                                                                    \
      }                                                                          \
    });                                                                          \
    return;                                                                      \
  }

#define HANDLE_INTERP(interp, align_corners, input_requires_grad)           \
  case interp: {                                                            \
    switch (static_cast<GridSamplerPadding>(padding_mode)) {                \
      HANDLE_CASE(interp, GridSamplerPadding::Zeros, align_corners, input_requires_grad);        \
      HANDLE_CASE(interp, GridSamplerPadding::Border, align_corners, input_requires_grad);       \
      HANDLE_CASE(interp, GridSamplerPadding::Reflection, align_corners, input_requires_grad);   \
    }                                                                       \
    return;                                                                 \
  }

  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, input.scalar_type(), "grid_sampler_2d_backward_cpu_kernel_impl", [&] {
    auto gGrid_acc = grad_grid.accessor<scalar_t, 4>();
    auto inp_acc = input.accessor<scalar_t, 4>();
    auto grid_acc = grid.accessor<scalar_t, 4>();
    auto gOut_acc = grad_output.accessor<scalar_t, 4>();
    if (input_requires_grad) {
      auto gInp_acc = grad_input.accessor<scalar_t, 4>();
      if (align_corners) {
        switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
          HANDLE_INTERP(GridSamplerInterpolation::Bilinear, true, true);
          HANDLE_INTERP(GridSamplerInterpolation::Nearest, true, true);
          HANDLE_INTERP(GridSamplerInterpolation::Bicubic, true, true);
        }
      } else {
        switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
          HANDLE_INTERP(GridSamplerInterpolation::Bilinear, false, true);
          HANDLE_INTERP(GridSamplerInterpolation::Nearest, false, true);
          HANDLE_INTERP(GridSamplerInterpolation::Bicubic, false, true);
        }
      }
    } else {
      if (align_corners) {
        switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
          HANDLE_INTERP(GridSamplerInterpolation::Bilinear, true, false);
          HANDLE_INTERP(GridSamplerInterpolation::Nearest, true, false);
          HANDLE_INTERP(GridSamplerInterpolation::Bicubic, true, false);
        }
      } else {
        switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
          HANDLE_INTERP(GridSamplerInterpolation::Bilinear, false, false);
          HANDLE_INTERP(GridSamplerInterpolation::Nearest, false, false);
          HANDLE_INTERP(GridSamplerInterpolation::Bicubic, false, false);
        }
      }

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

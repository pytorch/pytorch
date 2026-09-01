#pragma once
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/native/GridSamplerUtils.h>

namespace at::native {

using detail::GridSamplerInterpolation;
using detail::GridSamplerPadding;

// Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
// where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
// if align_corners: -1 and +1 get sent to the centers of the corner pixels
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// if not align_corners: -1 and +1 get sent to the image edges
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template <typename scalar_t>
__forceinline__ __device__
scalar_t grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1.f) * size - 1) / 2;
  }
}

// grid_sampler_unnormalize_set_grad works the same as grid_sampler_unnormalize
// except that it also returns the `d output / d input` via pointer argument
// `grad_in`.
// This is useful in the backward pass of grid_sampler.
template <typename scalar_t>
__forceinline__ __device__
scalar_t grid_sampler_unnormalize_set_grad(scalar_t coord, int size,
                                           bool align_corners, scalar_t *grad_in) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    *grad_in = static_cast<scalar_t>(size - 1) / 2;
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    *grad_in = static_cast<scalar_t>(size) / 2;
    return ((coord + 1.f) * size - 1) / 2;
  }
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
__forceinline__ __device__
scalar_t clip_coordinates(scalar_t in, int clip_limit) {
  return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
}

// clip_coordinates_set_grad works similarly to clip_coordinates except that
// it also returns the `d output / d input` via pointer argument `grad_in`.
// This is useful in the backward pass of grid_sampler.
template <typename scalar_t>
__forceinline__ __device__
scalar_t clip_coordinates_set_grad(scalar_t in, int clip_limit, scalar_t *grad_in) {
  // Note that it is important for the gradient calculation that borders
  // are considered out of bounds.
  if (in <= static_cast<scalar_t>(0)) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  } else {
    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
    if (in >= max) {
      *grad_in = static_cast<scalar_t>(0);
      return max;
    } else {
      *grad_in = static_cast<scalar_t>(1);
      return in;
    }
  }
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
__forceinline__ __device__
scalar_t reflect_coordinates(scalar_t in, int twice_low, int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = ::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = ::fmod(in, span);
  int flips = static_cast<int>(::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

// reflect_coordinates_set_grad works similarly to reflect_coordinates except
// that it also returns the `d output / d input` via pointer argument
// `grad_in`.
// This is useful in the backward pass of grid_sampler.
template <typename scalar_t>
__forceinline__ __device__
scalar_t reflect_coordinates_set_grad(scalar_t in, int twice_low, int twice_high,
                                      scalar_t *grad_in) {
  if (twice_low == twice_high) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  }
  int grad_in_mult_;
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = in - min;
  if (in < static_cast<scalar_t>(0)) {
    grad_in_mult_ = -1;
    in = -in;
  } else {
    grad_in_mult_ = 1;
  }
  // `fmod` returns same sign as `in`, which is positive after the `if` above.
  scalar_t extra = ::fmod(in, span);
  int flips = static_cast<int>(::floor(in / span));
  if (flips % 2 == 0) {
    *grad_in = static_cast<scalar_t>(grad_in_mult_);
    return extra + min;
  } else {
    *grad_in = static_cast<scalar_t>(-grad_in_mult_);
    return span - extra + min;
  }
}

template<typename scalar_t>
__forceinline__ __device__
scalar_t safe_downgrade_to_int_range(scalar_t x){
  // -100.0 does not have special meaning. This is just to make sure
  // it's not within_bounds_2d or within_bounds_3d, and does not cause
  // undefined behavior. See #35506.
  if (x > INT_MAX-1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}

template<typename scalar_t>
__forceinline__ __device__
scalar_t compute_coordinates(scalar_t coord, int size,
                             GridSamplerPadding padding_mode,
                             bool align_corners) {
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2*(size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2*size - 1);
    }
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
__forceinline__ __device__
scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int size,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  coord = compute_coordinates(coord, size, padding_mode, align_corners);
  return coord;
}

// grid_sampler_compute_source_index_set_grad works similarly to
// grid_sampler_compute_source_index except that it also returns the
// `d output / d input` via pointer argument `grad_in`.
// This is useful in the backward pass of grid_sampler.
template <typename scalar_t>
__forceinline__ __device__
scalar_t grid_sampler_compute_source_index_set_grad(
    scalar_t coord,
    int size,
    GridSamplerPadding padding_mode,
    bool align_corners,
    scalar_t *grad_in) {
  scalar_t grad_clip, grad_refl;
  coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_in);
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates_set_grad(coord, 0, 2*(size - 1), &grad_refl);
    } else {
      coord = reflect_coordinates_set_grad(coord, -1, 2*size - 1, &grad_refl);
    }
    // clip coordinates to image borders
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_refl * grad_clip;
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

__forceinline__ __device__
bool within_bounds_2d(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

__forceinline__ __device__
bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

template<typename scalar_t>
__forceinline__ __device__
scalar_t get_value_bounded(
    const scalar_t *data, scalar_t x, scalar_t y, int W, int H, int sW, int sH,
    GridSamplerPadding padding_mode,
    bool align_corners) {

  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);

  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);

  if (within_bounds_2d(iy, ix, H, W)) {
    return data[iy * sH + ix * sW];
  }
  return static_cast<scalar_t>(0);
}

template<typename scalar_t, typename index_t>
__forceinline__ __device__
void safe_add_2d(scalar_t *data, int h, int w,
                 int sH, int sW, int H, int W,
                 scalar_t delta,
                 const index_t NC_offset,
                 const index_t memory_span) {
  if (within_bounds_2d(h, w, H, W)) {
    fastAtomicAdd(data,
                  NC_offset + h * sH + w * sW,
                  memory_span,
                  delta,
                  true);
  }
}

template<typename scalar_t, typename index_t>
__forceinline__ __device__
void safe_add_3d(scalar_t *data, int d, int h, int w,
                 int sD, int sH, int sW, int D, int H, int W,
                 scalar_t delta,
                 const index_t NC_offset,
                 const index_t memory_span) {
  if (within_bounds_3d(d, h, w, D, H, W)) {
    fastAtomicAdd(data,
                  NC_offset + d * sD + h * sH + w * sW,
                  memory_span,
                  delta,
                  true);
  }
}

template<typename scalar_t, typename index_t>
__forceinline__ __device__
void add_value_bounded(
    scalar_t* data, scalar_t x, scalar_t y, int W, int H, int sW, int sH,
    scalar_t delta,
    GridSamplerPadding padding_mode,
    bool align_corners,
    const index_t NC_offset,
    const index_t memory_span) {

  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);

  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);

  safe_add_2d(data, iy, ix, sH, sW, H, W, delta, NC_offset, memory_span);
}

// Calculate the differential of the cubic convolution, i.e. `d coeff / d x`
template<typename scalar_t>
__forceinline__ __device__
void get_cubic_coefficients_grad(
    scalar_t coeffs[4],
    scalar_t t) {

  // Must be the same as forward calculation in
  // aten/src/ATen/native/cuda/UpSample.cuh:get_cubic_upsample_coefficients
  scalar_t A = -0.75;

  scalar_t x;
  x = -1 - t;  // 1 < x = |-1 - tx| < 2
  coeffs[0] = (-3 * A * x - 10 * A ) * x - 8 * A;
  x = -t;     // x = |0 - tx| <= 1
  coeffs[1] = (-3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 1 - t;  // x = |1 - tx| <= 1
  coeffs[2] = (3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 2 - t;  // 1 < x = |2 - tx| < 2
  coeffs[3] = (3 * A * x - 10 * A) * x + 8 * A;
}


// A non-finite pixel coordinate behaves as a very far finite one, identically
// on both devices and in every mode: a NaN far to the left, an infinity far on
// its own side. min/max and fmod order non-finites differently per device, so
// the mapping happens before any padding arithmetic.
template <typename scalar_t>
__forceinline__ __device__
scalar_t nonfinite_to_far(scalar_t coordinate) {
  if (::isnan(coordinate)) {
    return static_cast<scalar_t>(-100.0);
  }
  if (::isinf(static_cast<double>(coordinate))) {
    return coordinate > 0 ? static_cast<scalar_t>(1.0e15)
                          : static_cast<scalar_t>(-1.0e15);
  }
  return coordinate;
}

// The pixel route's source index: the padding applied directly in pixel
// units, defined for any coordinate. Border reaches its near edge from any
// magnitude, reflection keeps its phase through fmod with no integer
// conversion in the way, and zeros padding, which bounds nothing, guards the
// later integer casts with an out-of-volume sentinel.
template <typename scalar_t, typename index_t>
__forceinline__ __device__
scalar_t pixel_source_index(scalar_t x, index_t size,
                            GridSamplerPadding padding_mode,
                            bool align_corners) {
  x = nonfinite_to_far(x);
  if (padding_mode == GridSamplerPadding::Border) {
    x = ::min(static_cast<scalar_t>(size - 1),
              ::max(x, static_cast<scalar_t>(0)));
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    const scalar_t twice_low = align_corners ? 0 : -1;
    const scalar_t twice_high =
        static_cast<scalar_t>(2) * static_cast<scalar_t>(size) -
        (align_corners ? 2 : 1);
    if (twice_low == twice_high) {
      x = 0;
    } else {
      const scalar_t low = twice_low / 2;
      const scalar_t span = (twice_high - twice_low) / 2;
      const scalar_t in = ::fabs(x - low);
      const scalar_t extra = ::fmod(in, span);
      const bool odd = ::fmod(::floor(in / span), static_cast<scalar_t>(2)) != 0;
      x = odd ? span - extra + low : extra + low;
    }
    x = ::min(static_cast<scalar_t>(size - 1),
              ::max(x, static_cast<scalar_t>(0)));
  } else if (!(x > static_cast<scalar_t>(INT_MIN) &&
               x < static_cast<scalar_t>(INT_MAX - 1))) {
    // the kernels index in int on the 32-bit path, so the zeros arm, which
    // bounds nothing, guards that conversion
    x = static_cast<scalar_t>(-100.0);
  }
  return x;
}

// The set_grad twin: the padding's own derivative, one where the mapping is
// locally the identity, zero where border clips, the reflection's sign where
// it folds.
template <typename scalar_t, typename index_t>
__forceinline__ __device__
scalar_t pixel_source_index_set_grad(scalar_t x, index_t size,
                                     GridSamplerPadding padding_mode,
                                     bool align_corners, scalar_t* grad_in) {
  x = nonfinite_to_far(x);
  *grad_in = static_cast<scalar_t>(1);
  if (padding_mode == GridSamplerPadding::Border) {
    if (x < static_cast<scalar_t>(0) ||
        x > static_cast<scalar_t>(size - 1)) {
      *grad_in = static_cast<scalar_t>(0);
    }
    x = ::min(static_cast<scalar_t>(size - 1),
              ::max(x, static_cast<scalar_t>(0)));
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    const scalar_t twice_low = align_corners ? 0 : -1;
    const scalar_t twice_high =
        static_cast<scalar_t>(2) * static_cast<scalar_t>(size) -
        (align_corners ? 2 : 1);
    if (twice_low == twice_high) {
      x = 0;
      *grad_in = static_cast<scalar_t>(0);
    } else {
      const scalar_t low = twice_low / 2;
      const scalar_t span = (twice_high - twice_low) / 2;
      const scalar_t shifted = x - low;
      const scalar_t sign =
          shifted < 0 ? static_cast<scalar_t>(-1) : static_cast<scalar_t>(1);
      const scalar_t in = ::fabs(shifted);
      const scalar_t extra = ::fmod(in, span);
      const bool odd = ::fmod(::floor(in / span), static_cast<scalar_t>(2)) != 0;
      x = odd ? span - extra + low : extra + low;
      *grad_in = odd ? -sign : sign;
    }
    if (x < static_cast<scalar_t>(0) ||
        x > static_cast<scalar_t>(size - 1)) {
      *grad_in = static_cast<scalar_t>(0);
    }
    x = ::min(static_cast<scalar_t>(size - 1),
              ::max(x, static_cast<scalar_t>(0)));
  } else if (!(x > static_cast<scalar_t>(INT_MIN) &&
               x < static_cast<scalar_t>(INT_MAX - 1))) {
    // the kernels index in int on the 32-bit path, so the zeros arm, which
    // bounds nothing, guards that conversion
    x = static_cast<scalar_t>(-100.0);
  }
  return x;
}

// grid_sampler_unnormalize with the extent kept in index_t, for the kernels
// that index with int64_t. Each integer quantity is formed in index_t and
// converted where the int-taking helper converts it, so the two agree wherever
// that helper is defined, including for an extent a scalar_t cannot represent
// exactly. The copies
// exist so the kernels that share grid_sampler_unnormalize keep the code they
// generate today.
template <typename scalar_t, typename index_t>
__forceinline__ __device__
scalar_t grid_sampler_unnormalize_sized(scalar_t coord, index_t size,
                                        bool align_corners) {
  if (align_corners) {
    return ((coord + 1) / 2) * static_cast<scalar_t>(size - 1);
  } else {
    return ((coord + 1) * static_cast<scalar_t>(size) - 1) / 2;
  }
}

template <typename scalar_t, typename index_t>
__forceinline__ __device__
scalar_t grid_sampler_unnormalize_set_grad_sized(scalar_t coord, index_t size,
                                                 bool align_corners,
                                                 scalar_t* grad_in) {
  if (align_corners) {
    *grad_in = static_cast<scalar_t>(size - 1) / 2;
    return ((coord + 1) / 2) * static_cast<scalar_t>(size - 1);
  } else {
    *grad_in = static_cast<scalar_t>(size) / 2;
    return ((coord + 1) * static_cast<scalar_t>(size) - 1) / 2;
  }
}

// compute_coordinates with the extent kept in index_t: the tricubic kernels
// index with index_t, and narrowing the extent to int before the padding would
// fold a dimension past INT_MAX onto the wrong voxel. The reflection parity is
// taken with fmod so no float ever converts to an integer type, and no
// downgrade clips a valid position past INT_MAX: the caller's comparison gate
// decides before any cast. The bounds it forms are the ones compute_coordinates
// forms, for every extent; the parity and the missing downgrade are the two
// deliberate departures named above.
template <typename scalar_t, typename index_t>
__forceinline__ __device__
scalar_t compute_coordinates_sized(scalar_t coord, index_t size,
                                   GridSamplerPadding padding_mode,
                                   bool align_corners) {
  if (padding_mode == GridSamplerPadding::Border) {
    coord = ::min(static_cast<scalar_t>(size - 1),
                  ::max(coord, static_cast<scalar_t>(0)));
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect_coordinates takes twice_low and twice_high as integers and halves
    // their difference. Halving what it doubles reaches the same two bounds with
    // one conversion of an extent, which is exact where doubling in scalar_t is
    // not, and leaves nothing that could overflow the index type.
    const scalar_t low =
        align_corners ? static_cast<scalar_t>(0) : static_cast<scalar_t>(-0.5);
    const scalar_t span = static_cast<scalar_t>(align_corners ? size - 1 : size);
    if (span == 0) {
      coord = 0;
    } else {
      const scalar_t in = ::fabs(coord - low);
      const scalar_t extra = ::fmod(in, span);
      const bool odd = ::fmod(::floor(in / span), static_cast<scalar_t>(2)) != 0;
      coord = odd ? span - extra + low : extra + low;
    }
    coord = ::min(static_cast<scalar_t>(size - 1),
                  ::max(coord, static_cast<scalar_t>(0)));
  }
  return coord;
}

// The Keys coefficients with the coefficient as an argument, in the same
// expression order as get_cubic_upsampling_coefficients, so a = -0.75
// reproduces it bit for bit. The pixel route's helpers; the normalized route
// keeps the historical fixed-coefficient ones.
template<typename opmath_t>
__forceinline__ __device__
void get_cubic_coefficients_poly(opmath_t coeffs[4], opmath_t t, opmath_t a) {
  opmath_t x1 = t;
  coeffs[0] = cubic_convolution2<opmath_t>(x1 + 1.0, a);
  coeffs[1] = cubic_convolution1<opmath_t>(x1, a);
  opmath_t x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<opmath_t>(x2, a);
  coeffs[3] = cubic_convolution2<opmath_t>(x2 + 1.0, a);
}

template<typename opmath_t>
__forceinline__ __device__
void get_cubic_coefficients_a(opmath_t coeffs[4], opmath_t t, opmath_t a) {
  // At an integer location the weights are exactly [0, 1, 0, 0] for every a;
  // the polynomial evaluation only lands there when a's small multiples are
  // exactly representable, so the identity is taken outright.
  if (t == static_cast<opmath_t>(0)) {
    coeffs[0] = 0;
    coeffs[1] = 1;
    coeffs[2] = 0;
    coeffs[3] = 0;
    return;
  }
  get_cubic_coefficients_poly<opmath_t>(coeffs, t, a);
}

template<typename opmath_t>
__forceinline__ __device__
void get_cubic_coefficients_grad_a(opmath_t coeffs[4], opmath_t t, opmath_t a) {
  opmath_t x;
  x = -1 - t;
  coeffs[0] = (-3 * a * x - 10 * a) * x - 8 * a;
  x = -t;
  coeffs[1] = (-3 * (a + 2) * x - 2 * (a + 3)) * x;
  x = 1 - t;
  coeffs[2] = (3 * (a + 2) * x - 2 * (a + 3)) * x;
  x = 2 - t;
  coeffs[3] = (3 * a * x - 10 * a) * x + 8 * a;
}

// The device twin of resolve_cubic_taps in ATen/native/GridSampler.cpp. The
// normalized route's taps: the historical fixed-coefficient helpers, exactly
// the arithmetic it always ran.
template <typename coord_t, typename opmath_t, typename index_t>
__forceinline__ __device__
void resolve_cubic_taps(
    coord_t coord,
    index_t size,
    GridSamplerPadding padding_mode,
    bool align_corners,
    opmath_t coeffs[4],
    opmath_t* coeffs_grad,
    index_t indices[4]) {
  const coord_t base = ::floor(coord);
  const opmath_t t = static_cast<opmath_t>(coord - base);
  get_cubic_upsampling_coefficients<opmath_t>(coeffs, t);
  if (coeffs_grad != nullptr) {
    get_cubic_coefficients_grad<opmath_t>(coeffs_grad, t);
  }
  #pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    // the comparison decides, not the cast: a coordinate that is not
    // finite fails both sides, where converting it is undefined
    const coord_t tap = compute_coordinates_sized(
        base - 1 + i, size, padding_mode, align_corners);
    indices[i] = (tap >= 0 && tap < static_cast<coord_t>(size))
        ? static_cast<index_t>(tap)
        : static_cast<index_t>(-1);
  }
}

// `coord_t` places the taps (double when the grid is double), `opmath_t` is the
// payload's accumulate type the coefficients are blended in.
template<typename coord_t, typename opmath_t, typename index_t>
__forceinline__ __device__
void resolve_cubic_taps(
    coord_t coord,
    index_t size,
    GridSamplerPadding padding_mode,
    bool align_corners,
    opmath_t a,
    opmath_t coeffs[4],
    opmath_t* coeffs_grad,
    index_t indices[4]) {
  const coord_t base = ::floor(coord);
  const opmath_t t = static_cast<opmath_t>(coord - base);
  get_cubic_coefficients_a<opmath_t>(coeffs, t, a);
  if (coeffs_grad != nullptr) {
    get_cubic_coefficients_grad_a<opmath_t>(coeffs_grad, t, a);
  }
  #pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    // the comparison decides, not the cast: a coordinate that is not
    // finite fails both sides, where converting it is undefined
    const coord_t tap = compute_coordinates_sized(
        base - 1 + i, size, padding_mode, align_corners);
    indices[i] = (tap >= 0 && tap < static_cast<coord_t>(size))
        ? static_cast<index_t>(tap)
        : static_cast<index_t>(-1);
  }
}

}  // namespace at::native

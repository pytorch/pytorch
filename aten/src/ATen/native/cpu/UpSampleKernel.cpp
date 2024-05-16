#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <ATen/native/cpu/UpSampleKernelAVXAntialias.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/ones.h>
#endif

namespace at::native {
namespace {

using scale_t = std::vector<c10::optional<double>>;

// TODO: this file could benefit from a global renaming of its functions /
// classes and terms, as well as from adding more comments. In particular:
// - It's not obvious that despite their names (and the file name), all these
//   kernels don't just do upsampling: they do general interpolation, i.e. they
//   also all support downscaling.
// - the term "horizontal" or "within dims" or "contiguous dim" refers to the
//   last dimension.
//   It's not specific to 2D images and applies to 3D (and 1D??) inputs as well.
//   Similarly "vertical" or "across dims" refers to all dims that aren't the
//   last one. In other kernels these are also referred to as "zero-stride" and
//   "non-zero-stride" - we should unify all this.
// - the terms "zero-stride" and "non-zero strides" refer to the weights and
//   indices, not to the contiguity of input or output
// - It's not always clear which kernel is vectorized and which one isn't.
// - The functions like _use_vectorized_kernel_cond() should be renamed and
//   their description updated, because they're not the only "fork" in the
//   code-path where a choice is made between a vectorized kernel vs a
//   non-vectorized one. See e.g. upsample_bilinear2d_kernel_impl() where we
//   already make a similar check, before the one in
//   _use_vectorized_kernel_cond().
// - It's not always clear which code is part of a "separable interpolation"
//   code-path.
// - Some names need to be more specific. For example
//   "cpu_upsample_generic_aa()" looks like a super generic name, but the function
//   is instead fairly specific - we need to make that clearer.
// - Some functions have a "aa" suffix but it doesn't mean that they only
//   support antialias. Some of them also support antialias=False now.
// - Various comments are outdated. Case in point: the one just below about the
//   `Interpolate` struct being used for cpu_upsample_linear:
//   cpu_upsample_linear doesn't exist anymore, and these structs are used for
//   various modes, *not* just linear.
// - It'd be useful to document how interpolation works in general, and in particular state explicitly:
//   - that the weights and indices across a given dimension are the same for
//     all pixels (hence the benefit of pre-computing them)
//   - that it can be "separated", i.e. we can do the horizontal pass and the
//     vertical pass independently (and that some kernels are written this way,
//     while some aren't.)
// - we can probably remove the template over index_t, because it's always
//   hard-coded as int64_t


// Helper structs and methods for cpu_upsample_linear
//
// Interpolation methods that used below are separable, and as such we can compute the interpolation
// independently per dimension in a recursive way. Please, refer to #10482 for more context.
//
// Interpolation structure to compute output value in n-dimensional case.
// - recursively compute interpolated output for each dimension
// - we rely a lot on compiler's code optimization such that implemented operations
//   can be automatically factorized and vectorized using SSE and AVX2
template <int n, typename scalar_t, typename opmath_t, typename index_t, int interp_size>
struct Interpolate {
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      opmath_t wts = *(scalar_t*)&data[1][i * strides[1]];
      opmath_t t = Interpolate<n - 1, scalar_t, opmath_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);
      opmath_t output = t * wts;
      for (const auto j : c10::irange(1, interp_size)) {
        ids = *(index_t*)&data[2 * j + 0][i * strides[2 * j + 0]];
        wts = *(scalar_t*)&data[2 * j + 1][i * strides[2 * j + 1]];
        t = Interpolate<n - 1, scalar_t, opmath_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);
        output += t * wts;
      }
      return output;
  }
};

template <typename scalar_t, typename opmath_t, typename index_t, int interp_size>
struct Interpolate<1, scalar_t, opmath_t, index_t, interp_size> {
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      opmath_t wts = *(scalar_t*)&data[1][i * strides[1]];
      opmath_t t = *(scalar_t *)&src[ids];
      opmath_t output = t * wts;
      for (const auto j : c10::irange(1, interp_size)) {
        ids = *(index_t*)&data[2 * j + 0][i * strides[2 * j + 0]];
        wts = *(scalar_t*)&data[2 * j + 1][i * strides[2 * j + 1]];
        t = *(scalar_t *)&src[ids];
        output += t * wts;
      }
      return output;
    }
};

template <int n, typename scalar_t, typename opmath_t, typename index_t>
struct Interpolate<n, scalar_t, opmath_t, index_t, 1> {
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      return Interpolate<n - 1, scalar_t, opmath_t, index_t, 1>::eval(src + ids, &data[2], &strides[2], i);
  }
};

template <typename scalar_t, typename opmath_t, typename index_t>
struct Interpolate<1, scalar_t, opmath_t, index_t, 1> {
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      return *(scalar_t *)&src[ids];
    }
};

// There is an unexpected 2x slowdown for upsample_trilinear3d channels_first
// for both 1 and 6 threads. We have to specialize this case as below:
// Once the issue is fixed we can keep generic implementation and remove:
// struct Interpolate<n, scalar_t, index_t, 2> and
// struct Interpolate<1, scalar_t, index_t, 2>
template <int n, typename scalar_t, typename opmath_t, typename index_t>
struct Interpolate<n, scalar_t, opmath_t, index_t, 2> {
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        index_t i0 = *(index_t*)&data[0][i * strides[0]];
        index_t i1 = *(index_t*)&data[2][i * strides[2]];
        opmath_t w0 = *(scalar_t *)&data[1][i * strides[1]];
        opmath_t w1 = *(scalar_t *)&data[3][i * strides[3]];

        opmath_t t0 = Interpolate<n - 1, scalar_t, opmath_t, index_t, 2>::eval(src + i0, &data[4], &strides[4], i);
        opmath_t t1 = Interpolate<n - 1, scalar_t, opmath_t, index_t, 2>::eval(src + i1, &data[4], &strides[4], i);

        return t0 * w0 + t1 * w1;
  }
};

template <typename scalar_t, typename opmath_t, typename index_t>
struct Interpolate<1, scalar_t, opmath_t, index_t, 2> {
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        index_t i0 = *(index_t*)&data[0][i * strides[0]];
        index_t i1 = *(index_t*)&data[2][i * strides[2]];
        opmath_t w0 = *(scalar_t *)&data[1][i * strides[1]];
        opmath_t w1 = *(scalar_t *)&data[3][i * strides[3]];
        opmath_t t0 = *(scalar_t *)&src[i0];
        opmath_t t1 = *(scalar_t *)&src[i1];
        return t0 * w0 + t1 * w1;
    }
};

template <int n, typename scalar_t, typename index_t, int interp_size>
static inline scalar_t interpolate(char* src, char** data, const int64_t* strides, int64_t i) {
  using opmath_t = at::opmath_type<scalar_t>;
  return Interpolate<n, scalar_t, opmath_t, index_t, interp_size>::eval(src, data, strides, i);
}

template <typename scalar_t, typename index_t>
static inline scalar_t interpolate_aa_single_dim_zero_strides(
    char* src,
    char** data,
    const index_t ids_stride) {
  const index_t ids_min = *(index_t*)&data[0][0];
  const index_t ids_size = *(index_t*)&data[1][0];

  char* src_min = src + ids_min;

  scalar_t t = *(scalar_t*)&src_min[0];
  index_t wts_idx = *(index_t*)&data[4][0];
  scalar_t* wts_ptr = (scalar_t*)&data[3][wts_idx];
  scalar_t wts = wts_ptr[0];

  scalar_t output = t * wts;
  for (const auto j : c10::irange(1, ids_size)) {
    wts = wts_ptr[j];
    t = *(scalar_t*)&src_min[j * ids_stride];
    output += t * wts;
  }
  return output;
}

template <typename scalar_t, typename index_t>
static inline scalar_t interpolate_aa_single_dim(
    char* src,
    char** data,
    const int64_t* strides,
    int64_t i,
    const index_t ids_stride) {
  index_t ids_min = *(index_t*)&data[0][i * strides[0]];
  index_t ids_size = *(index_t*)&data[1][i * strides[1]];

  char* src_min = src + ids_min;

  scalar_t t = *(scalar_t*)&src_min[0];
  index_t wts_idx = *(index_t*)&data[4][i * strides[4]];
  scalar_t* wts_ptr = (scalar_t*)&data[3][wts_idx];
  scalar_t wts = wts_ptr[0];

  scalar_t output = t * wts;
  for (const auto j : c10::irange(1, ids_size)) {
    wts = wts_ptr[j];
    t = *(scalar_t*)&src_min[j * ids_stride];
    output += t * wts;
  }
  return output;
}

template<int m>
static inline bool is_zero_stride(const int64_t* strides) {
  bool output = strides[0] == 0;
  for (const auto i : c10::irange(1, m)) {
    output &= (strides[i] == 0);
  }
  return output;
}

template <typename scalar_t, typename index_t, int interp_size>
static inline bool is_contiguous_stride(const int64_t* strides) {
  bool output = (strides[0] == sizeof(index_t)) && (strides[1] == sizeof(scalar_t));
  for (int i=2; i<2 * interp_size; i+=2) {
    output &= (strides[i] == sizeof(index_t)) && (strides[i + 1] == sizeof(scalar_t));
  }
  return output;
}

// Helper class to recursively check if all input strides corresponding to interpolated dimensions
// are equal zero except on a single dimension.
//
// Inputs: array of strides of size N, non_zero_stride_dim which can be -1, 0, 1, 2, ...
//   if non_zero_stride_dim, we check that all strides are equal zero, otherwise
//   4 strides corresponding to the strides for index_0, weight_0, index_1 and weight_1 for non_zero_stride_dim
//   dimension should be non zero.
//
// Unit check of the recursion is to verify whether 4 strides for one interpolated dimension are either zero,
// see method is_zero_stride, or (sizeof(index_t), sizeof(scalar_t), sizeof(index_t), sizeof(scalar_t)), see
// method is_contiguous_stride.
//
// In practice, we have the following cases:
// - for ND, float32, channel first, strides are
//         dimN-1,              dim1,           dim0
//         i0, w0, i1, w1, ..., i0, w0, i1, w1, i0, w0, i1, w1
// strides=(0,  0,  0,  0, ...,  0,  0,  0,  0,  4,  4,  4,  4)
//
// if size dim0 is 1 then its strides are 0 and dim1 strides are equal 4
//
// - for ND, float32, channel last, strides are
//         dimN-1,         dimN-2,             dim0
//         i0, w0, i1, w1, i0, w0, i1, w1, ... i0, w0, i1, w1
// strides=(0,  0,  0,  0,  0,  0,  0,  0, ..., 0,  0,  0,  0)
//
// Using these methods we can hint the compiler to factorize constant indices and weights
// in cpu_upsample_linear method
template <int N, int non_zero_stride_dim, typename scalar_t, typename index_t, int interp_size>
struct CheckAlmostAllZeroStrides {
  static inline bool eval(const int64_t* strides) {
    // N is dim index: N -> dim0, N-1 -> dim1, ...
    // non_zero_stride_dim should be out_dims - dim
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool output;
    if (N == non_zero_stride_dim) {
      output = is_contiguous_stride<scalar_t, index_t, interp_size>(strides);
    } else {
      output = is_zero_stride<2 * interp_size>(strides);
    }
    return output &&
      CheckAlmostAllZeroStrides<N - 1, non_zero_stride_dim, scalar_t, index_t, interp_size>::eval(
        &strides[2 * interp_size]);
  }
};

template <int non_zero_stride_dim, typename scalar_t, typename index_t, int interp_size>
struct CheckAlmostAllZeroStrides<0, non_zero_stride_dim, scalar_t, index_t, interp_size> {
  static inline bool eval(const int64_t* /*strides*/) {
    return true;
  }
};

template <int n, int s, typename scalar_t, typename index_t, int interp_size>
static inline bool check_almost_all_zero_stride(const int64_t* strides) {
  return CheckAlmostAllZeroStrides<n, s, scalar_t, index_t, interp_size>::eval(strides);
}

// Helper method to compute interpolation for nearest, linear, cubic modes
template <typename scalar_t, typename index_t, int out_ndims, int interp_size>
static inline void basic_loop(char** data, const int64_t* strides, int64_t n) {
  char* dst = data[0];
  char* src = data[1];
  for (const auto i : c10::irange(n)) {
    *(scalar_t*)&dst[i * strides[0]] = interpolate<out_ndims, scalar_t, index_t, interp_size>(
        src + i * strides[1], &data[2], &strides[2], i);
  }
}

template <typename scalar_t>
static inline void basic_loop_aa_vertical(
    char** data,
    const int64_t* strides,
    int64_t n,
    unsigned int weights_precision) {
  char* dst = data[0];
  char* src = data[1];
  // index stride is constant for the given dimension
  const int64_t ids_stride = *(int64_t*)&data[2 + 2][0];

  for (const auto i : c10::irange(n)) {
    *(scalar_t*)&dst[i * strides[0]] =
        interpolate_aa_single_dim_zero_strides<scalar_t, int64_t>(
            src + i * strides[1], &data[2], ids_stride);
  }
}

template <>
inline void basic_loop_aa_vertical<uint8_t>(
    char** data,
    const int64_t* strides,
    int64_t n,
    unsigned int weights_precision) {
  // See Note [ Weights computation for uint8_t and multiplication trick ]
  char* dst = data[0];
  char* src = data[1];

  // index stride is constant for the given dimension
  const int64_t ids_stride = *(int64_t*)&data[2 + 2][0];
  const int64_t ids_size = *(int64_t*)&data[2 + 1][0];
  const int64_t ids_min = *(int64_t*)&data[2 + 0][0];

  int64_t i = 0;

  for (; i<n; i++) {

    char* src_min = src + i * strides[1] + ids_min;

    uint8_t t = *(uint8_t*)&src_min[0];
    int64_t wts_idx = *(int64_t*)&data[2 + 4][0];
    int16_t* wts_ptr = (int16_t*)&data[2 + 3][wts_idx];
    int16_t wts = wts_ptr[0];

    // Intermediate computations are using integer type
    int output = 1 << (weights_precision - 1);  // accounts for the +0.5 part
    output += t * wts;
    for (const auto j : c10::irange(1, ids_size)) {
      wts = wts_ptr[j];
      t = *(uint8_t*)&src_min[j * ids_stride];
      output += t * wts;
    }
    *(uint8_t*)&dst[i * strides[0]] = (uint8_t)std::clamp(output >> weights_precision, 0, 255);
  }
}

template <typename scalar_t>
static inline void basic_loop_aa_horizontal(
    char** data,
    const int64_t* strides,
    int64_t n,
    unsigned int weights_precision) {
  char* dst = data[0];
  char* src = data[1];
  // index stride is constant for the given dimension
  const int64_t ids_stride = *(int64_t*)&data[2 + 2][0];

  if (strides[1] == 0) {
    for (const auto i : c10::irange(n)) {
      *(scalar_t*)&dst[i * strides[0]] =
          interpolate_aa_single_dim<scalar_t, int64_t>(
              src, &data[2], &strides[2], i, ids_stride);
    }
  } else {
    for (const auto i : c10::irange(n)) {
      *(scalar_t*)&dst[i * strides[0]] =
          interpolate_aa_single_dim<scalar_t, int64_t>(
              src + i * strides[1], &data[2], &strides[2], i, ids_stride);
    }
  }
}

template <>
inline void basic_loop_aa_horizontal<uint8_t>(
    char** data,
    const int64_t* strides,
    int64_t n,
    unsigned int weights_precision) {
  // See Note [ Weights computation for uint8_t and multiplication trick ]
  char* dst = data[0];
  char* src = data[1];
  // index stride is constant for the given dimension
  const int64_t ids_stride = *(int64_t*)&data[2 + 2][0];

  int64_t i = 0;

  // Here we are implementing data interpolation within the same line (vs between the lines)
  // output[x, y] = input[xmin[x], y] * W[x] + input[xmin[x] + 1, y] * W[x + 1] + ... + input[xmin[x] + xsize, y] * W[x + xsize]

  for (; i<n; i++) {

    int64_t ids_min = *(int64_t*)&data[2 + 0][i * strides[2 + 0]];
    int64_t ids_size = *(int64_t*)&data[2 + 1][i * strides[2 + 1]];

    char* src_min = src + i * strides[1] + ids_min;

    uint8_t t = *(uint8_t*)&src_min[0];
    int64_t wts_idx = *(int64_t*)&data[2 + 4][i * strides[2 + 4]];
    int16_t* wts_ptr = (int16_t*)&data[2 + 3][wts_idx];
    int16_t wts = wts_ptr[0];

    // Intermediate computations are using integer type
    int output = 1 << (weights_precision - 1);  // accounts for the +0.5 part
    output += t * wts;
    for (const auto j : c10::irange(1, ids_size)) {
      wts = wts_ptr[j];
      t = *(uint8_t*)&src_min[j * ids_stride];
      output += t * wts;
    }
    *(uint8_t*)&dst[i * strides[0]] = (uint8_t)std::clamp(output >> weights_precision, 0, 255);
  }
}

// Generic upsampling computation method using TensorIterator for Nd case.
// Supports: nearest, linear, cubic modes with interp_size template argument: 1, 2, 4
//
// Single loop function for 1d, 2d and 3d cases and modes
// For N dimensions, output value up to Di dimension can be computed as
//
// output_i[a] = interpolate(output_{i+1}[a], w_{i+1}[a], output_{i+1}[a+1], w_{i+1}[a+1], ...)
// with
// output_DN[a] = interpolate(input_DN[a], w_DN[a], input_DN[a+1], w_DN[a+1], ...)
// and i - dimension index and a - linear index for spatial coordinates
//
// The recursive call is implemented with InterpLinear struct using template for
// the loop unrolling on compile time.
template <typename scalar_t, int out_ndims, int interp_size>
void cpu_upsample_generic(at::TensorIterator& iter)
{
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    // special-cases to let the compiler apply compile-time input-specific optimizations
    if ((strides[0] == sizeof(scalar_t) && (strides[1] == 0) &&
        // NOLINTNEXTLINE(bugprone-branch-clone)
        check_almost_all_zero_stride<out_ndims, 1, scalar_t, int64_t, interp_size>(&strides[2]))) {
      // contiguous channels-first case
      basic_loop<scalar_t, int64_t, out_ndims, interp_size>(data, strides, n);
    } else if ((strides[0] == sizeof(scalar_t) && (strides[1] == sizeof(scalar_t)) &&
               check_almost_all_zero_stride<out_ndims, -1, scalar_t, int64_t, interp_size>(&strides[2]))) {
      // contiguous channels-last case
      basic_loop<scalar_t, int64_t, out_ndims, interp_size>(data, strides, n);
    } else {
      // fallback
      basic_loop<scalar_t, int64_t, out_ndims, interp_size>(data, strides, n);
    }
  };
  iter.for_each(loop);
}

template <typename scalar_t, typename scale_type, nearest_idx_fn_t nearest_idx_fn>
void cpu_upsample_nearest_channels_last(
    const Tensor& output_,
    const Tensor& input_,
    const scale_type& scales) {
  TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
              " for `output` but got dtype ", output_.dtype());

  auto input_sizes = input_.sizes().vec();
  auto output_sizes = output_.sizes().vec();
  auto ndim = input_sizes.size();
  TORCH_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

  auto channels_last_memory_format = ndim == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
  auto input = input_.contiguous(channels_last_memory_format);
  auto output = output_.contiguous(channels_last_memory_format);

  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t num_batches =  input_sizes[0];
  int64_t channels =  input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];
  int64_t numel = output.numel();

  TORCH_CHECK(channels > 0, "expected input and output channels greater than 0 but got ", channels);

  using Vec = vec::Vectorized<scalar_t>;
  auto copy = [](scalar_t* out, const scalar_t* in, int64_t size) {
    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      Vec out_vec = Vec::loadu(in + d);
      out_vec.store(out + d);
    }
    for (; d < size; d++) {
      out[d] = in[d];
    }
  };

  auto loop2d = [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, num_batches, oh, output_height, ow, output_width);

    for (const auto i : c10::irange(begin, end)) {
      int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[0]);
      int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[1]);
      scalar_t* output_ptr = output_data + i * channels;
      const scalar_t* input_ptr = input_data + n * input_height * input_width * channels +
          ih * input_width * channels + iw * channels;
      copy(output_ptr, input_ptr, channels);
      data_index_step(n, num_batches, oh, output_height, ow, output_width);
    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, num_batches, od, output_depth, oh, output_height, ow, output_width);

    for (const auto i : c10::irange(begin, end)) {
      int64_t id = nearest_idx_fn(od, input_depth, output_depth, scales[0]);
      int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[1]);
      int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[2]);
      scalar_t* output_ptr = output_data + i * channels;
      const scalar_t* input_ptr = input_data + n * input_depth * input_height * input_width * channels +
          id * input_height * input_width * channels +
          ih * input_width * channels + iw * channels;
      copy(output_ptr, input_ptr, channels);
      data_index_step(n, num_batches, od, output_depth, oh, output_height, ow, output_width);
    }
  };

  if (ndim == 4) {
    // upsample nearest 2d
    at::parallel_for(0, numel / channels, at::internal::GRAIN_SIZE / channels, loop2d);
  } else {
    // upsample nearest 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, numel / channels, at::internal::GRAIN_SIZE / channels, loop3d);
  }

  if (!output_.is_contiguous(channels_last_memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t, typename accscalar_t>
inline VecType<scalar_t> interpolate(const scalar_t* t, accscalar_t w) {
  return VecType<scalar_t>::loadu(t) * VecType<scalar_t>(w);
}

template <typename scalar_t, typename accscalar_t, typename... Args>
inline VecType<scalar_t> interpolate(const scalar_t* t, accscalar_t w, Args... args) {
  return VecType<scalar_t>::loadu(t) * VecType<scalar_t>(w) + interpolate(args...);
}

template <typename scalar_t, typename scale_type>
void cpu_upsample_linear_channels_last(
    const Tensor& output_,
    const Tensor& input_,
    bool align_corners,
    const scale_type& scales) {
  TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
              " for `output` but got dtype ", output_.dtype());

  auto input_sizes = input_.sizes().vec();
  auto output_sizes = output_.sizes().vec();
  auto ndim = input_sizes.size();
  TORCH_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

  auto channels_last_memory_format = ndim == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
  auto input = input_.contiguous(channels_last_memory_format);
  auto output = output_.contiguous(channels_last_memory_format);

  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t num_batches =  input_sizes[0];
  int64_t channels =  input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  TORCH_CHECK(channels > 0, "expected input and output channels greater than 0 but got ", channels);
  int64_t output_slice_size = output_depth * output_height * output_width * channels;

  using opmath_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<scalar_t>;
  auto loop2d = [&](int64_t begin, int64_t end) {
    const auto height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[0]);
    const auto width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[1]);

    auto input_indexr = [=](int64_t n, int64_t h, int64_t w) {
      return input_data + n * input_height * input_width * channels +
          h * input_width * channels + w * channels;
    };

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ih0, ih1, iw0, iw1;
    opmath_t h0lambda, h1lambda, w0lambda, w1lambda;
    for (const auto n : c10::irange(begin, end)) {
      for (const auto oh : c10::irange(output_height)) {
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        for (const auto ow : c10::irange(output_width)) {
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);

          scalar_t* out = output_data + n * output_slice_size +
              oh * output_width * channels + ow * channels;
          const scalar_t* i00 = input_indexr(n, ih0, iw0);
          const scalar_t* i01 = input_indexr(n, ih0, iw1);
          const scalar_t* i10 = input_indexr(n, ih1, iw0);
          const scalar_t* i11 = input_indexr(n, ih1, iw1);
          opmath_t w00 = h0lambda * w0lambda;
          opmath_t w01 = h0lambda * w1lambda;
          opmath_t w10 = h1lambda * w0lambda;
          opmath_t w11 = h1lambda * w1lambda;

          int64_t size = channels;
          int64_t d = 0;
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            auto out_vec = interpolate(i00 + d, w00, i01 + d, w01, i10 + d, w10, i11 + d, w11);
            out_vec.store(out + d);
          }
          for (; d < size; d++) {
            out[d] = i00[d] * w00 + i01[d] * w01 + i10[d] * w10 + i11[d] * w11;
          }
        }
      }
    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    const auto depth_scale = area_pixel_compute_scale<opmath_t>(
        input_depth, output_depth, align_corners, scales[0]);
    const auto height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[1]);
    const auto width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[2]);

    auto input_indexr = [=](int64_t n, int64_t d, int64_t h, int64_t w) {
      return input_data + n * input_depth * input_height * input_width * channels +
          d * input_height * input_width * channels +
          h * input_width * channels + w * channels;
    };

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t id0, id1, ih0, ih1, iw0, iw1;
    opmath_t d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
    for (const auto n : c10::irange(begin, end)) {
      for (const auto od : c10::irange(output_depth)) {
        compute_source_index_and_lambda(
            id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
        for (const auto oh : c10::irange(output_height)) {
          compute_source_index_and_lambda(
              ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
          for (const auto ow : c10::irange(output_width)) {
            compute_source_index_and_lambda(
                iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);

            scalar_t* out = output_data + n * output_slice_size +
                od * output_height * output_width * channels +
                oh * output_width * channels + ow * channels;
            const scalar_t* i000 = input_indexr(n, id0, ih0, iw0);
            const scalar_t* i001 = input_indexr(n, id0, ih0, iw1);
            const scalar_t* i010 = input_indexr(n, id0, ih1, iw0);
            const scalar_t* i011 = input_indexr(n, id0, ih1, iw1);
            const scalar_t* i100 = input_indexr(n, id1, ih0, iw0);
            const scalar_t* i101 = input_indexr(n, id1, ih0, iw1);
            const scalar_t* i110 = input_indexr(n, id1, ih1, iw0);
            const scalar_t* i111 = input_indexr(n, id1, ih1, iw1);
            opmath_t w000 = d0lambda * h0lambda * w0lambda;
            opmath_t w001 = d0lambda * h0lambda * w1lambda;
            opmath_t w010 = d0lambda * h1lambda * w0lambda;
            opmath_t w011 = d0lambda * h1lambda * w1lambda;
            opmath_t w100 = d1lambda * h0lambda * w0lambda;
            opmath_t w101 = d1lambda * h0lambda * w1lambda;
            opmath_t w110 = d1lambda * h1lambda * w0lambda;
            opmath_t w111 = d1lambda * h1lambda * w1lambda;

            int64_t size = channels;
            int64_t d = 0;
            for (; d < size - (size % Vec::size()); d += Vec::size()) {
              auto out_vec = interpolate(
                  i000 + d, w000, i001 + d, w001, i010 + d, w010, i011 + d, w011,
                  i100 + d, w100, i101 + d, w101, i110 + d, w110, i111 + d, w111);
              out_vec.store(out + d);
            }
            for (; d < size; d++) {
              out[d] =
                  i000[d] * w000 + i001[d] * w001 + i010[d] * w010 + i011[d] * w011 +
                  i100[d] * w100 + i101[d] * w101 + i110[d] * w110 + i111[d] * w111;
            }
          }
        }
      }
    }
  };

  if (ndim == 4) {
    // upsample nearest 2d
    at::parallel_for(0, num_batches, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    // upsample nearest 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, num_batches, at::internal::GRAIN_SIZE / output_slice_size / 8, loop3d);
  }

  if (!output_.is_contiguous(channels_last_memory_format)) {
    output_.copy_(output);
  }
}

// Helper structs to use with upsample_generic_Nd_kernel_impl
struct HelperInterpBase {

  static inline void init_indices_weights(
    at::ScalarType output_type,
    std::vector<Tensor> & output, int64_t output_size, int64_t ndims,
    int64_t reshape_dim, int interp_size
  ) {

    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    for (const auto j C10_UNUSED : c10::irange(interp_size)) {
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
      output.emplace_back(empty(new_shape, CPU(output_type)));
    }
  }

  // This is a helper function for _compute_index_ranges_weights method that computes
  // source two int64 scalars index min and size and a list weights (of size max_interp_size)
  // for interpolation with antialiasing=true mode. It returns the maximal weights value
  template <typename scalar_t, typename aa_filter_fn_t>
  static inline scalar_t _compute_indices_min_size_weights_aa(
    const int64_t i, const int64_t input_size, const scalar_t scale, const scalar_t support,
    scalar_t* wt_ptr, const int64_t max_interp_size, aa_filter_fn_t filter_fn,
    int64_t& xmin, int64_t& xsize
  ) {

    scalar_t center = scale * (i + 0.5);
    scalar_t total_w = 0.0;
    scalar_t invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
    xmin = std::max(
        static_cast<int64_t>(center - support + 0.5), static_cast<int64_t>(0));
    xsize = std::min(
        static_cast<int64_t>(center + support + 0.5), input_size) - xmin;
    // There are rare cases when due to precision xsize can be larger than max_interp_size by one.
    // We have to clip the value
    xsize = std::clamp(xsize, static_cast<int64_t>(0), max_interp_size);

    int64_t j = 0;
    for (; j < xsize; j++) {
      scalar_t w = filter_fn((j + xmin - center + 0.5) * invscale);
      wt_ptr[j] = w;
      total_w += w;
    }

    scalar_t wt_max = 0.0;
    if (total_w != 0.0) {
      for (j = 0; j < xsize; j++) {
        wt_ptr[j] /= total_w;
        wt_max = std::max(wt_max, wt_ptr[j]);
      }
    }

    for (; j < max_interp_size; j++) {
      wt_ptr[j] = static_cast<scalar_t>(0.0);
    }
    return wt_max;
  }

  // This is a helper function for _compute_index_ranges_weights method that computes
  // source two int64 scalars index min and size and a list weights (of size max_interp_size)
  // for interpolation with antialiasing=false mode. It returns the maximal weights value.
  // This function is templated with scalar_t for type of scale and weights but is only used for
  // bilinear/bicubic modes on uint8 input and antialiasing=false (in this case scalar_t is double).
  // For float input types we are using upsample_generic_Nd_kernel_impl and compute_indices_weights methods
  template <typename scalar_t, typename aa_filter_fn_t>
  static inline scalar_t _compute_indices_min_size_weights(
    const int64_t i, const int64_t input_size, const scalar_t scale,
    scalar_t* wt_ptr, const int64_t max_interp_size, aa_filter_fn_t filter_fn,
    bool align_corners, int64_t& index_min, int64_t& index_size
  ) {
    // Notes. We do not use opmath_t in this method as f16 and other smaller float types are not routed here.
    // Typical usage of this method is with scalar_t = double when computing indices and weights for uint8 input
    // The code below partly adapts indices and lambda computation from compute_indices_weights method and
    // index_min/index_size from _compute_indices_min_size_weights_aa

    bool cubic = max_interp_size > 2;
    const auto real_input_index = area_pixel_compute_source_index<scalar_t>(
        scale, i, align_corners, /*cubic=*/cubic);

    scalar_t lambda;
    int64_t input_index;
    guard_index_and_lambda(real_input_index, input_size, input_index, lambda);

    const auto support = static_cast<int64_t>(max_interp_size * 0.5);
    const auto unbound_index_min = input_index - support + 1;
    const auto unbound_index_max = input_index + support + 1;
    index_min = std::max(unbound_index_min, static_cast<int64_t>(0));
    index_size = std::min(unbound_index_max, input_size) - index_min;
    // There are rare cases when due to precision xsize can be larger than max_interp_size by one.
    // We have to clip the value
    index_size = std::clamp(index_size, static_cast<int64_t>(0), max_interp_size);

    // Below the weights are computed using filter_fn and accumulating values for indices being out of bounds
    // For example, for bicubic mode for output index i = 0, we have input_index = -1,
    // then we have unbound_index_min = -2 and unbound_index_max = 1 => unbounded input indices are [-2, -1, 0, 1] and
    // valid input indices will be [0, 1]
    // For unbounded input indices we compute four non-zero weights values [w0, w1, w2, w3] and as only two weights can
    // be used with valid input indcies, we accumulate values in the following way: [w0 + w1 + w2, w3, 0.0, 0.0]
    // This is equivalent to the float path which would compute indices as [0, 0, 0, 1] and weights as [w0, w1, w2, s3].
    // A similar accumulation should done for unbounded indices larger than input size.
    auto w_index = 0;
    scalar_t wt_max = 0.0;
    for (const auto j : c10::irange(max_interp_size)) {
      // initialize weights value as we will accumulate below
      wt_ptr[j] = 0.0;

      scalar_t w = filter_fn(static_cast<scalar_t>(j + 1 - support) - lambda);
      if (unbound_index_min + j <= 0) {
        w_index = 0;
      } else if (unbound_index_min + j >= input_size - 1) {
        w_index = index_size - 1;
      }
      wt_ptr[w_index] += w;
      wt_max = std::max(wt_max, wt_ptr[w_index]);
      w_index++;
    }

    return wt_max;
  }

  // Note [ Support for antialias=False as a subcase of antialias=True ]
  // This function was originally written with the hard assumption that
  // antialias=True and it was later extended to support antialias=False.
  // The only difference between aa and no-aa is in how the
  // weights and indices are computed (and their number). In aa their number is
  // variable but with no-aa, they're fixed to interp_size. The same "filters"
  // can be used otherwise. HOWEVER, support for antialias=False here may not be
  // optimally optimized: the code assumes an arbitrary number of weights and
  // indices, but this can be optimized further when aa=False since we know
  // their actual dimensions.
  template <typename scalar_t, typename aa_filter_fn_t, int weight_index_stride=sizeof(scalar_t)>
  static inline std::tuple<std::vector<Tensor>, int, scalar_t> _compute_index_ranges_weights(
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims,
    int64_t reshape_dim, scalar_t scale,
    int interp_size, aa_filter_fn_t aa_filter_fn, bool antialias, bool align_corners
  ) {

    std::vector<Tensor> output;

    scalar_t support;
    int max_interp_size;
    if (antialias) {
        support = (scale >= 1.0) ? (interp_size * 0.5) * scale : interp_size * 0.5;
        max_interp_size = (int) std::ceil(support) * 2 + 1;
    } else {
        support = interp_size * 0.5;
        max_interp_size = interp_size;
    }

    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    // Bounds approach as in PIL: xmin/xmax
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));

    {
      // Weights
      new_shape[reshape_dim] = output_size * max_interp_size;
      auto wts = empty(new_shape, CPU(c10::CppTypeToScalarType<scalar_t>()));
      auto strides = wts.strides().vec();
      strides[reshape_dim] = 0;
      new_shape[reshape_dim] = output_size;
      wts = wts.as_strided(new_shape, strides);
      output.emplace_back(wts);
      // Weights indices
      output.emplace_back(
          empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
    }

    int64_t* idx_ptr_xmin = output[0].data_ptr<int64_t>();
    int64_t* idx_ptr_size = output[1].data_ptr<int64_t>();
    int64_t* idx_ptr_stride = output[2].data_ptr<int64_t>();
    scalar_t* wt_ptr = output[3].data_ptr<scalar_t>();
    int64_t* wt_idx_ptr = output[4].data_ptr<int64_t>();

    scalar_t wt_max = 0.0;
    for (const auto i : c10::irange(output_size)) {
      int64_t xmin, xsize;
      scalar_t wt_max_i;
      if (antialias) {
        wt_max_i = HelperInterpBase::_compute_indices_min_size_weights_aa(
            i,
            input_size,
            scale,
            support,
            wt_ptr + i * max_interp_size,
            max_interp_size,
            aa_filter_fn,
            xmin,
            xsize);
      } else {
        wt_max_i = HelperInterpBase::_compute_indices_min_size_weights(
            i,
            input_size,
            scale,
            wt_ptr + i * max_interp_size,
            max_interp_size,
            aa_filter_fn,
            align_corners,
            xmin,
            xsize);
      }
      wt_max = std::max(wt_max, wt_max_i);

      idx_ptr_xmin[i] = xmin * stride;
      idx_ptr_size[i] = xsize;
      idx_ptr_stride[i] = stride;
      wt_idx_ptr[i] = i * max_interp_size * weight_index_stride;
    }
    return {output, max_interp_size, wt_max};
  }

  /*
  NOTE [ Weights computation for uint8_t and multiplication trick ]
  When the input/output dtype is uint8_t, we still compute the interpolation
  weights as double, but then convert them to int16 via some conversion logic
  detailed below. This allows us to compute all interpolation operation (sum of
  multiplications) as ints instead of floats. The result is converted back into
  uint8 in basic_loop_aa_horizontal<uint8_t> (and vertical)

  In essence the idea is to avoid a multiplication between a float (the
  weight) and an int (the pixel value) and instead run a multiplication between
  2 ints:

  ```py
  COEF_PREC = 16

  def mul(a:float, b:int) -> Tuple[float, int]:
    # return a * b, round(a * b)
    actual = a * b

    assert a > 0  # I'm lazy
    int_a = floor(0.5 + a * (1 << COEF_PREC))
    with_trick = ((int_a * b) + (1 << (COEF_PREC - 1))) >> COEF_PREC

    return actual, with_trick  # round(actual) == with_trick!!
  ```

  Here's how it works:
  N == COEFF_PREC
  1 << N == 2**N
  floor(0.5 + x) == round(x)

  So the operation is something like

  int_a = round(a * 2**N)  -- let's just say it's `a * 2**N` for simplicity

  res = ((int_a * b) + (1 << (N - 1))) >> N
      = ((a * 2**N * b + 2**(N - 1)) / 2**N
      = a * b + 0.5
      = round(a * b)
      = what we wanted
  */
  template <typename aa_filter_fn_t>
  static inline std::tuple<std::vector<Tensor>, int, unsigned int> _compute_index_ranges_int16_weights(
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims,
    int64_t reshape_dim, bool align_corners, const c10::optional<double> opt_scale,
    int interp_size, aa_filter_fn_t aa_filter_fn, bool antialias, bool align_i32=false
  ) {

    double scale = area_pixel_compute_scale<double>(
        input_size, output_size, align_corners, opt_scale);

    std::vector<Tensor> indices_weights;
    double wt_max;
    std::tie(indices_weights, interp_size, wt_max) = HelperInterpBase::_compute_index_ranges_weights<double, aa_filter_fn_t, sizeof(int16_t)>(
        input_size, output_size, stride, ndims, reshape_dim, scale, interp_size, aa_filter_fn, antialias, align_corners);

    // Rescale float weights to int16 and compute weights precision
    auto weights_f64 = indices_weights[3];
    double * data_f64 = weights_f64.data_ptr<double>();

    unsigned int weights_precision = 0;
    for (weights_precision = 0; weights_precision < 22; ++weights_precision) {
        int next_value = (int) (0.5 + wt_max * (1 << (weights_precision + 1)));
        if (next_value >= (1 << 15))
            break;
    }

    // Rescale float values to int16
    int16_t * data_i16 = (int16_t *) data_f64;
    auto aligned_interp_size = interp_size;

    if (align_i32) {
      // We should respect int32 alignment as we will load int16 data as int32
      // See ImagingResampleHorizontalConvolution8u4x, mmk0 = _mm256_set1_epi32(*(int32_t*)&k[x]);
      // compute aligned_interp_size = nearest pair value to interp_size
      while (aligned_interp_size % sizeof(int32_t) != 0) {
        aligned_interp_size += 1;
      }
      // assert that we wont go out of bounds
      TORCH_INTERNAL_ASSERT(aligned_interp_size * sizeof(int16_t) < interp_size * sizeof(double));
    }

    for (const auto j : c10::irange(output_size)) {
      for (const auto k : c10::irange(interp_size)) {
        double v = data_f64[j * interp_size + k] * (1 << weights_precision);
        data_i16[j * aligned_interp_size + k] = (v < 0) ? (int) (-0.5 + v) : (int) (0.5 + v);
      }
    }

    return {indices_weights, aligned_interp_size, weights_precision};
  }
};

struct HelperInterpNearest : public HelperInterpBase {
  // This structure implements outdated and buggy method to compute indices
  // for nearest neighbours interpolation
  // We keep this structure for BC and consider as deprecated.
  // See HelperInterpNearestExact as replacement

  static const int interp_size = 1;

  static inline void init_indices_weights(
    at::ScalarType output_type,
    std::vector<Tensor> & output, int64_t output_size, int64_t ndims,
    int64_t reshape_dim, int interp_size
  ) {
    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    for (const auto j C10_UNUSED : c10::irange(interp_size)) {
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
      // Defines weights for consistency, but not used
      output.emplace_back(at::ones(new_shape, CPU(output_type)));
    }
  }

  // Compute nearest mode indices and weights for each interpolated dimension
  // indices_weights = {
  //      {indices_0, 1.0, },  // dim -n
  //      {indices_0, 1.0, },  // dim -(n-1)
  //      ...
  //      {indices_0, 1.0, },  // dim -1
  // }
  // Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to
  // fit input/output tensors.
  // Indices are already containing the strides to optimize the computations
  static inline std::vector<Tensor> compute_indices_weights(
    at::ScalarType scalar_type,
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims,
    int64_t reshape_dim, bool align_corners, const c10::optional<double> opt_scale
  ) {

    TORCH_INTERNAL_ASSERT(!align_corners);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<Tensor> output;
    HelperInterpNearest::init_indices_weights(
      scalar_type, output, output_size, ndims, reshape_dim, HelperInterpNearest::interp_size);

    AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, scalar_type, "compute_indices_weights_nearest", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        opmath_t scale = area_pixel_compute_scale<opmath_t>(input_size, output_size, align_corners, opt_scale);

        auto input_index_ptr = output[0].data_ptr<int64_t>();
        int64_t input_index;

        // Indices are computed as following:
        // scale = 1.0 * isize / osize
        // index_f32 = (output_index) * scale
        // input_index = floor(index_f32)
        // Same as OpenCV INTER_NEAREST
        for (const auto i : c10::irange(output_size)) {
          const auto real_input_index =
              area_pixel_compute_source_index<opmath_t>(
                  scale, i, /*align_corners=*/true, /*cubic=*/false);
          input_index = static_cast<int64_t>(floorf(real_input_index));
          input_index_ptr[i] = static_cast<int64_t>(std::min(input_index, input_size - 1)) * stride;
        }
      }
    );
    return output;
  }

};

struct HelperInterpNearestExact : public HelperInterpNearest {

  // Compute nearest mode indices and weights for each interpolated dimension
  // indices_weights = {
  //      {indices_0, 1.0, },  // dim -n
  //      {indices_0, 1.0, },  // dim -(n-1)
  //      ...
  //      {indices_0, 1.0, },  // dim -1
  // }
  // Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to
  // fit input/output tensors.
  // Indices are already containing the strides to optimize the computations
  static inline std::vector<Tensor> compute_indices_weights(
    at::ScalarType scalar_type,
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims,
    int64_t reshape_dim, bool align_corners, const c10::optional<double> opt_scale
  ) {

    TORCH_INTERNAL_ASSERT(!align_corners);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<Tensor> output;
    HelperInterpNearest::init_indices_weights(
      scalar_type, output, output_size, ndims, reshape_dim, HelperInterpNearest::interp_size);

    AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, scalar_type, "compute_indices_weights_nearest", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        opmath_t scale = area_pixel_compute_scale<opmath_t>(input_size, output_size, align_corners, opt_scale);

        auto input_index_ptr = output[0].data_ptr<int64_t>();
        int64_t input_index;

        // Indices should be computed as following:
        // scale = 1.0 * isize / osize
        // index_f32 = (output_index + 0.5) * scale - 0.5
        // input_index = round(index_f32)
        // Same as Pillow and Scikit-Image/Scipy ndi.zoom
        for (const auto i : c10::irange(output_size)) {
          const auto real_input_index =
              area_pixel_compute_source_index<opmath_t>(
                  scale, i, /*align_corners=*/align_corners, /*cubic=*/false);
          input_index = static_cast<int64_t>(floorf(real_input_index + 0.5));
          input_index_ptr[i] = static_cast<int64_t>(std::min(input_index, input_size - 1)) * stride;
        }
      }
    );
    return output;
  }
};

struct HelperInterpLinear : public HelperInterpBase {

  static const int interp_size = 2;

  // Compute indices and weights for each interpolated dimension
  // indices_weights = {
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -n
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -(n-1)
  //      ...
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -1
  // }
  // Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to
  // fit input/output tensors.
  // Indices are already containing the strides to optimize the computations
  static inline std::vector<Tensor> compute_indices_weights(
    at::ScalarType scalar_type,
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims, int64_t reshape_dim,
    bool align_corners, const c10::optional<double> opt_scale
  ) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<Tensor> output;
    HelperInterpLinear::init_indices_weights(
      scalar_type, output, output_size, ndims, reshape_dim, HelperInterpLinear::interp_size);
    AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, scalar_type, "compute_indices_weights_linear", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        opmath_t scale = area_pixel_compute_scale<opmath_t>(input_size, output_size, align_corners, opt_scale);

        auto input_index0_ptr = output[0].data_ptr<int64_t>();
        auto lambda0_ptr = output[1].data_ptr<scalar_t>();
        auto input_index1_ptr = output[2].data_ptr<int64_t>();
        auto lambda1_ptr = output[3].data_ptr<scalar_t>();

        for (const auto i : c10::irange(output_size)) {

          compute_source_index_and_lambda<scalar_t, opmath_t>(
            input_index0_ptr[i], input_index1_ptr[i],
            lambda0_ptr[i], lambda1_ptr[i],
            scale, i, input_size, output_size, align_corners
          );
          // put stride into indices
          // index values correspond to input indices (0, 1, 2, 3, ...)
          // when multiplied by input stride, maximum possible value
          // input_size[dim-1] * input_size[dim-2] * ... for the given dimension.
          input_index0_ptr[i] *= stride;
          input_index1_ptr[i] *= stride;
        }
      }
    );
    return output;
  }

  // taken from
  // https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c#L20-L29
  template<typename scalar_t>
  static inline scalar_t aa_filter(scalar_t x) {
    x = std::abs(x);
    if (x < 1.0) {
      return 1.0 - x;
    }
    return 0.0;
  }

  static inline std::vector<Tensor> compute_index_ranges_weights(
    at::ScalarType scalar_type,
    int64_t input_size,
    int64_t output_size,
    int64_t stride,
    int64_t ndims,
    int64_t reshape_dim,
    bool align_corners,
    const c10::optional<double> opt_scale,
    bool antialias
  ) {

    std::vector<Tensor> indices_weights;
    AT_DISPATCH_FLOATING_TYPES(
      scalar_type, "compute_index_ranges_weights", [&] {

        scalar_t scale = area_pixel_compute_scale<scalar_t>(
            input_size, output_size, align_corners, opt_scale);

        auto interp_size = HelperInterpLinear::interp_size;

        indices_weights = std::get<0>(HelperInterpLinear::_compute_index_ranges_weights<scalar_t>(
            input_size,
            output_size,
            stride,
            ndims,
            reshape_dim,
            scale,
            interp_size,
            &HelperInterpLinear::aa_filter<scalar_t>,
            /*antialias=*/antialias,
            /*align_corners=*/align_corners));
      }
    );
    return indices_weights;
  }

  static inline std::tuple<std::vector<Tensor>, int, unsigned int> compute_index_ranges_int16_weights(
    int64_t input_size,
    int64_t output_size,
    int64_t stride,
    int64_t ndims,
    int64_t reshape_dim,
    bool align_corners,
    const c10::optional<double> opt_scale,
    bool antialias,
    bool align_i32=false
  ) {

    auto interp_size = HelperInterpLinear::interp_size;
    auto fn = HelperInterpLinear::aa_filter<double>;
    return HelperInterpLinear::_compute_index_ranges_int16_weights(
        input_size, output_size, stride, ndims, reshape_dim,
        align_corners, opt_scale, interp_size, fn, antialias, align_i32);
  }
};

struct HelperInterpCubic : public HelperInterpBase {

  static const int interp_size = 4;

  // Compute indices and weights for each interpolated dimension
  // indices_weights = {
  //      {indices_0, weights_0, indices_1, weights_1, ..., indices_3, weights_3},  // dim -n
  //      {indices_0, weights_0, indices_1, weights_1, ..., indices_3, weights_3},  // dim -(n-1)
  //      ...
  //      {indices_0, weights_0, indices_1, weights_1, ..., indices_3, weights_3},  // dim -1
  // }
  // Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to
  // fit input/output tensors.
  // Indices are already containing the strides to optimize the computations
  static inline std::vector<Tensor> compute_indices_weights(
    at::ScalarType scalar_type,
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims, int64_t reshape_dim,
    bool align_corners, const c10::optional<double> opt_scale
  ) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<Tensor> output;
    HelperInterpCubic::init_indices_weights(
      scalar_type, output, output_size, ndims, reshape_dim, HelperInterpCubic::interp_size);

    AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, scalar_type, "compute_indices_weights_cubic", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        opmath_t scale = area_pixel_compute_scale<opmath_t>(input_size, output_size, align_corners, opt_scale);

        int64_t input_index;
        int64_t zero = static_cast<int64_t>(0);
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        opmath_t coeffs[4];

        int64_t * idx_ptr;
        scalar_t * wt_ptr;
        for (const auto i : c10::irange(output_size)) {
          const auto real_input_index =
              area_pixel_compute_source_index<opmath_t>(
                  scale, i, align_corners, /*cubic=*/true);
          opmath_t lambda;
          guard_index_and_lambda(real_input_index, input_size, input_index, lambda);
          get_cubic_upsample_coefficients<opmath_t>(coeffs, lambda);

          for (const auto j : c10::irange(interp_size)) {
            idx_ptr = output[2 * j + 0].data_ptr<int64_t>();
            idx_ptr[i] = static_cast<int64_t>(std::max(std::min(input_index + j - 1, input_size - 1), zero)) * stride;
            wt_ptr = output[2 * j + 1].data_ptr<scalar_t>();
            wt_ptr[i] = coeffs[j];
          }
        }
      }
    );
    return output;
  }

  // taken from
  // https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c#L46-L62
  template<typename scalar_t, bool use_keys_cubic=true>
  static inline scalar_t aa_filter(scalar_t x) {
    // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    // a = -0.5 was proposed by R. Keys in "Cubic convolution interpolation for digital image processing"
    // We are using -0.5 for bicubic, antialiasing=true (compatibility with PIL)
    // and using -0.75 for bicubic, antialiasing=false (compatibility with Opencv)
    constexpr scalar_t a = use_keys_cubic ? -0.5 : -0.75;

    x = std::abs(x);
    if (x < 1.0) {
        return cubic_convolution1(x, a);
    }
    if (x < 2.0) {
        return cubic_convolution2(x, a);
    }
    return 0.0;
  }

  static inline std::vector<Tensor> compute_index_ranges_weights(
    at::ScalarType scalar_type,
    int64_t input_size,
    int64_t output_size,
    int64_t stride,
    int64_t ndims,
    int64_t reshape_dim,
    bool align_corners,
    const c10::optional<double> opt_scale,
    bool antialias
  ) {

    std::vector<Tensor> indices_weights;
    AT_DISPATCH_FLOATING_TYPES(
      scalar_type, "compute_index_ranges_weights", [&] {

        scalar_t scale = area_pixel_compute_scale<scalar_t>(
            input_size, output_size, align_corners, opt_scale);

        auto interp_size = HelperInterpCubic::interp_size;

        indices_weights = std::get<0>(HelperInterpCubic::_compute_index_ranges_weights<scalar_t>(
            input_size,
            output_size,
            stride,
            ndims,
            reshape_dim,
            scale,
            interp_size,
            &HelperInterpCubic::aa_filter<scalar_t>,
            /*antialias=*/antialias,
            /*align_corners=*/align_corners));
      }
    );
    return indices_weights;
  }

  static inline std::tuple<std::vector<Tensor>, int, unsigned int> compute_index_ranges_int16_weights(
    int64_t input_size,
    int64_t output_size,
    int64_t stride,
    int64_t ndims,
    int64_t reshape_dim,
    bool align_corners,
    const c10::optional<double> opt_scale,
    bool antialias,
    bool align_i32=false
  ) {

    auto interp_size = HelperInterpCubic::interp_size;
    // We have to use the -0.75 constant when aa is False so that this uint8
    // path is as close as possible to float results.
    auto fn = antialias ? HelperInterpCubic::aa_filter<double, true> : HelperInterpCubic::aa_filter<double, false>;
    return HelperInterpCubic::_compute_index_ranges_int16_weights(
        input_size, output_size, stride, ndims, reshape_dim,
        align_corners, opt_scale, interp_size, fn, antialias, align_i32);
  }

};

// Generic upsampling interpolation kernel for N-d case.
// Input is assumed to be like NCHW, NCL, NCKHW - interpolated spatial dimension
// are those from the end up to batch size N and number of channels C.
//
// Internally, it uses TensorIterator to optimize the computations.
// - out_ndims is the number of interpolated dims: 1, 2, 3
// - scale_type is template type for scales, typically c10::optional<double>
// - template<typename> class F is one of the above structs to compute indices and weights
template <int out_ndims, typename scale_type, class F>
void upsample_generic_Nd_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    const scale_type& scales) {


  // input can be NCHW, NCL or NCKHW
  auto shape = input.sizes().vec();
  auto strides = input.strides().vec();
  auto oshape = output.sizes();

  TORCH_INTERNAL_ASSERT(
    shape.size() == oshape.size() && shape.size() == 2 + out_ndims
  );
  TORCH_INTERNAL_ASSERT(strides.size() == 2 + out_ndims);

  for (const auto i : c10::irange(out_ndims)) {
    shape[i + 2] = oshape[i + 2];
    strides[i + 2] = 0;
  }
  auto restrided_input = input.as_strided(shape, strides);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<std::vector<Tensor>> indices_weights;

  constexpr int interp_size = F::interp_size;
  auto input_scalar_type = input.scalar_type();
  if ((interp_size == 1 && input_scalar_type == at::ScalarType::Byte)) {
    // nearest also supports uint8 tensor, but we have to use float
    // with compute_indices_weights
    input_scalar_type = at::ScalarType::Float;
  }

  for (const auto i : c10::irange(out_ndims)) {
    // NOLINTNEXTLINE(performance-inefficient-vector-operation)
    indices_weights.emplace_back(
      F::compute_indices_weights(
        input_scalar_type, input.size(i + 2), oshape[i + 2],
        input.stride(i + 2) * input.element_size(),
        input.dim(), i + 2, align_corners, scales[i]
      )
    );
  }

  TensorIteratorConfig config;
  config.check_all_same_dtype(false)
    .declare_static_dtype_and_device(input.scalar_type(), input.device())
    .add_output(output)
    .add_const_input(restrided_input);

  for (auto & idx_weight: indices_weights) {
    for (auto& tensor : idx_weight) {
      config.add_const_input(tensor);
    }
  }

  auto iter = config.build();

  if (interp_size > 1) {
    // Nearest also supports uint8 tensor, so need to handle it separately
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16, kHalf, iter.dtype(), "upsample_generic_Nd", [&] {
        // MSVC can not catch constexpr int interp_size here
        constexpr int mode = F::interp_size;
        cpu_upsample_generic<scalar_t, out_ndims, mode>(iter);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND3(kByte, kBFloat16, kHalf,
        iter.dtype(), "upsample_generic_Nd", [&] {
        constexpr int mode = F::interp_size;
        cpu_upsample_generic<scalar_t, out_ndims, mode>(iter);
    });
  }
}

template <typename scalar_t, bool is_horizontal>
void cpu_upsample_generic_aa(at::TensorIterator& iter, unsigned int weights_precision) {

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    if (is_horizontal) {

      // Strides are : X 0 | 8 8 8 0 8  (Channels first)
      // Strides are : X X | 0 0 0 0 0  (Channels last)
      // upsampling data within a contiguous dimension (aka horizontal resampling)
      if ((strides[0] == sizeof(scalar_t)) && (strides[1] == sizeof(scalar_t)) &&
          is_zero_stride<3 + 2>(&strides[2])) {
        // channels last case
        basic_loop_aa_horizontal<scalar_t>(
            data, strides, n, weights_precision);
      } else {
        basic_loop_aa_horizontal<scalar_t>(
            data, strides, n, weights_precision);
      }
    } else {
      // Strides are : X Y | 0 0 0 0 0 (Channels first)
      // Strides are : X X | 0 0 0 0 0 (Channels last)
      // upsampling data between contiguous dimensions (aka vertical resampling)
      if ((strides[0] == sizeof(scalar_t)) && (strides[1] == sizeof(scalar_t)) &&
          is_zero_stride<3 + 2>(&strides[2])) {
        basic_loop_aa_vertical<scalar_t>(
            data, strides, n, weights_precision);
      } else {
        basic_loop_aa_vertical<scalar_t>(
            data, strides, n, weights_precision);
      }
    }
  };

  iter.for_each(loop);
}

template <int out_ndims, typename scale_type, class F, bool is_horizontal>
void _separable_upsample_generic_Nd_kernel_impl_single_dim(
    const Tensor& output,
    const Tensor& input,
    int interp_dim,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {

  // input can be NCHW, NCL or NCKHW
  auto shape = input.sizes().vec();
  auto strides = input.strides().vec();
  auto oshape = output.sizes();

  TORCH_INTERNAL_ASSERT(
      shape.size() == oshape.size() && shape.size() == 2 + out_ndims);
  TORCH_INTERNAL_ASSERT(strides.size() == 2 + out_ndims);

  for (const auto i : c10::irange(out_ndims)) {
    shape[i + 2] = oshape[i + 2];
  }
  strides[interp_dim] = 0;
  auto restrided_input = input.as_strided(shape, strides);

  auto input_scalar_type = input.scalar_type();

  std::vector<Tensor> indices_weights;
  unsigned int weights_precision = 0;
  int unused;

  if (input_scalar_type == at::kByte) {
    // This is a special branch to provide uint8 dtype support for bilinear and bicubic modes only
    TORCH_INTERNAL_ASSERT(F::interp_size == 2 || F::interp_size == 4);
    std::tie(indices_weights, unused, weights_precision) =
      F::compute_index_ranges_int16_weights(
        input.size(interp_dim), oshape[interp_dim],
        input.stride(interp_dim) * input.element_size(),
        input.dim(), interp_dim, align_corners, scales[interp_dim - 2],
        antialias);
    TORCH_INTERNAL_ASSERT(weights_precision > 0);
  } else {
    indices_weights =
      F::compute_index_ranges_weights(
        input_scalar_type, input.size(interp_dim), oshape[interp_dim],
        input.stride(interp_dim) * input.element_size(),
        input.dim(), interp_dim, align_corners, scales[interp_dim - 2],
        antialias);
  }

  TensorIteratorConfig config;
  config.check_all_same_dtype(false)
      .declare_static_dtype_and_device(input.scalar_type(), input.device())
      .add_output(output)
      .add_const_input(restrided_input);

  for (auto& tensor : indices_weights) {
    config.add_const_input(tensor);
  }

  auto iter = config.build();

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Byte, iter.dtype(), "upsample_generic_Nd_aa", [&] {
        cpu_upsample_generic_aa<scalar_t, is_horizontal>(iter, weights_precision);
      });
}

// Generic separable upsampling interpolation kernel for N-d case with anti-aliasing.
// It also supports antialias=False iff
// (dtype == uint8 and mode in ("bilinear", "bicubic")): this is used as
// fallback in these settings when AVX isn't supported.
template <int out_ndims, typename scale_type, class F>
void separable_upsample_generic_Nd_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {

  auto output_shape = output.sizes();
  auto input_shape = input.sizes();
  auto temp_oshape = input_shape.vec();

  if (output_shape == input_shape) {
    output.copy_(input);
    return;
  }

  at::Tensor temp_output, temp_input = input;

  int interp_dim = 0;
  // Precompute the number of single dim resize method invocations
  // to avoid copying temporary buffer to output
  int num_single_dim_ops = 0;
  for (const auto i : c10::irange(out_ndims)) {
    interp_dim = 2 + out_ndims - 1 - i;
    if (output_shape[interp_dim] != input_shape[interp_dim]) {
      num_single_dim_ops += 1;
    }
  }

  // upsampling data within the contiguous dimension (aka horizontal resampling)
  interp_dim = 2 + out_ndims - 1;
  if (output_shape[interp_dim] != input_shape[interp_dim]) {

    num_single_dim_ops -= 1;
    if (num_single_dim_ops > 0) {
      temp_oshape[interp_dim] = output_shape[interp_dim];
      temp_output = at::empty(temp_oshape, input.options());
    } else {
      temp_output = output;
    }

    _separable_upsample_generic_Nd_kernel_impl_single_dim<
        out_ndims,
        scale_t,
        F,
        true>(
        temp_output, temp_input, interp_dim, align_corners, scales, antialias);
    temp_input = temp_output;
  }

  // upsampling data between contiguous dimensions (aka vertical resampling)
  for (const auto i : c10::irange(1, out_ndims)) {
    interp_dim = 2 + out_ndims - 1 - i;
    if (output_shape[interp_dim] != input_shape[interp_dim]) {

      num_single_dim_ops -= 1;
      if (num_single_dim_ops > 0) {
        temp_oshape[interp_dim] = output_shape[interp_dim];
        temp_output = at::empty(temp_oshape, input.options());
      } else {
        temp_output = output;
      }

      _separable_upsample_generic_Nd_kernel_impl_single_dim<
          out_ndims,
          scale_t,
          F,
          false>(
          temp_output, temp_input, interp_dim, align_corners, scales, antialias);
      temp_input = temp_output;
    }
  }
}

void upsample_nearest1d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    c10::optional<double> scales_w) {
  upsample_generic_Nd_kernel_impl<1, scale_t, HelperInterpNearest>(
      output, input, false, {scales_w});
}

void _upsample_nearest_exact1d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    c10::optional<double> scales_w) {
  upsample_generic_Nd_kernel_impl<1, scale_t, HelperInterpNearestExact>(
    output, input, false, {scales_w});
}

int _use_vectorized_kernel_cond_2d(
    const Tensor& output,
    const Tensor& input) {
      // This condition is used to know whether we should dispatch to a vectorized
      // kernel, or to the more general upsample_generic_Nd_kernel_impl(). For now,
      // the vectorized kernels are only optimized for channels_last and when C >= 4
      // (shape = NCHW). For a very wide range of use-cases (typically image or mask
      // resizing where we have C < 4), using upsample_generic_Nd_kernel_impl() is
      // actually faster. On top of that, benchmarks showed that this also depends on
      // the *output* size (output_H + output_W), for both upsampling and
      // downsampling. The current 128 threshold was determined through benchmarks.
      return ((input.is_contiguous(at::MemoryFormat::ChannelsLast)) && (input.size(1) > 3)) || ((output.size(-2) + output.size(-1)) <= 128);
}

int _use_vectorized_kernel_cond_3d(
    // Similar to _use_vectorized_kernel_cond_2d() but for 3d resampling (e.g. videos)
    // Note that unlike the 2d case, this is not subject to small output size
    // overhead - hence the absence of the 128 threshold in the condition.
    const Tensor& output,
    const Tensor& input) {
      return ((input.is_contiguous(at::MemoryFormat::ChannelsLast3d)) && (input.size(1) > 3));
}


void upsample_nearest2d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (_use_vectorized_kernel_cond_2d(output, input)) {
    AT_DISPATCH_FLOATING_TYPES_AND3(kByte, kBFloat16, kHalf,
        input.scalar_type(), "upsample_nearest2d_channels_last", [&] {
      cpu_upsample_nearest_channels_last<scalar_t, scale_t, nearest_idx>(output, input, {scales_h, scales_w});
    });
  } else {
    upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpNearest>(
      output, input, false, {scales_h, scales_w});
  }
}

void _upsample_nearest_exact2d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (_use_vectorized_kernel_cond_2d(output, input)) {
    AT_DISPATCH_FLOATING_TYPES_AND3(kByte, kBFloat16, kHalf, input.scalar_type(), "upsample_nearest2d_channels_last", [&] {
      cpu_upsample_nearest_channels_last<scalar_t, scale_t, nearest_exact_idx>(output, input, {scales_h, scales_w});
    });
  } else {
    upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpNearestExact>(
      output, input, false, {scales_h, scales_w});
  }
}

void upsample_nearest3d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (_use_vectorized_kernel_cond_3d(output, input)) {
    AT_DISPATCH_FLOATING_TYPES_AND3(kByte, kBFloat16, kHalf,
        input.scalar_type(), "upsample_nearest3d_channels_last", [&] {
      cpu_upsample_nearest_channels_last<scalar_t, scale_t, nearest_idx>(output, input, {scales_d, scales_h, scales_w});
    });
  } else {
    upsample_generic_Nd_kernel_impl<3, scale_t, HelperInterpNearest>(
      output, input, false, {scales_d, scales_h, scales_w});
  }
}

void _upsample_nearest_exact3d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (_use_vectorized_kernel_cond_3d(output, input)) {
    AT_DISPATCH_FLOATING_TYPES_AND3(kByte, kBFloat16, kHalf, input.scalar_type(), "upsample_nearest3d_channels_last", [&] {
      cpu_upsample_nearest_channels_last<scalar_t, scale_t, nearest_exact_idx>(output, input, {scales_d, scales_h, scales_w});
    });
  } else {
    upsample_generic_Nd_kernel_impl<3, scale_t, HelperInterpNearestExact>(
      output, input, false, {scales_d, scales_h, scales_w});
  }
}

void upsample_linear1d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_w) {
  upsample_generic_Nd_kernel_impl<1, scale_t, HelperInterpLinear>(
    output, input, align_corners, {scales_w});
}


void upsample_bilinear2d_kernel_impl_float(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {

  // See note above about _use_vectorized_kernel_cond_2d(output, input). The extra cond is present
  // because benchmarks showed that with only 1 thread, images (C == 3) were
  // slightly faster with the vectorized kernel than with the generic one.
  // That's not the case for masks though (C == 1), which strongly benefit from
  // using the generic kernel.
  if ((_use_vectorized_kernel_cond_2d(output, input)) || (at::get_num_threads() == 1 && input.size(1) == 3)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "upsample_bilinear2d_channels_last", [&] {
      cpu_upsample_linear_channels_last<scalar_t, scale_t>(output, input, align_corners, {scales_h, scales_w});
    });
  } else {
    upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpLinear>(
      output, input, align_corners, {scales_h, scales_w});
  }
}

void upsample_bilinear2d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {

  if (input.dtype() == at::kByte){
    #ifdef CPU_CAPABILITY_AVX2
      if (input.size(1) <= 4) {
        upsample_avx_bilinear_bicubic_uint8<scale_t, HelperInterpLinear>(input,
          output, align_corners, {scales_h, scales_w},
          /*antialias=*/false);
      } else {
        separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpLinear>(
          output, input, align_corners, {scales_h, scales_w},
          /*antialias=*/false);
      }
    #else  // CPU_CAPABILITY_AVX2
      separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpLinear>(
        output, input, align_corners, {scales_h, scales_w},
        /*antialias=*/false);
    #endif  // CPU_CAPABILITY_AVX2
  } else {
    upsample_bilinear2d_kernel_impl_float(output, input, align_corners, scales_h, scales_w);
  }
}


void upsample_bilinear2d_aa_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
#ifdef CPU_CAPABILITY_AVX2
  if (input.dtype() == at::kByte && input.size(1) <= 4) {
    upsample_avx_bilinear_bicubic_uint8<scale_t, HelperInterpLinear>(
      input, output, align_corners, {scales_h, scales_w},
      /*antialias=*/true);
  } else {
    separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpLinear>(
        output, input, align_corners, {scales_h, scales_w},
        /*antialias=*/true);
  }
#else // CPU_CAPABILITY_AVX2
  separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpLinear>(
      output, input, align_corners, {scales_h, scales_w},
      /*antialias=*/true);
#endif // CPU_CAPABILITY_AVX2
}

void upsample_trilinear3d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if ((_use_vectorized_kernel_cond_3d(output, input))) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "upsample_trilinear3d_channels_last", [&] {
      cpu_upsample_linear_channels_last<scalar_t, scale_t>(output, input, align_corners, {scales_d, scales_h, scales_w});
    });
  } else {
    upsample_generic_Nd_kernel_impl<3, scale_t, HelperInterpLinear>(
      output, input, align_corners, {scales_d, scales_h, scales_w});
  }
}

void upsample_bicubic2d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {

  if (input.dtype() == at::kByte){
    #ifdef CPU_CAPABILITY_AVX2
      if (input.size(1) <= 4) {
        upsample_avx_bilinear_bicubic_uint8<scale_t, HelperInterpCubic>(input,
          output, align_corners, {scales_h, scales_w},
          /*antialias=*/false);
      } else {
        separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpCubic>(
          output, input, align_corners, {scales_h, scales_w},
          /*antialias=*/false);
      }
    #else  // CPU_CAPABILITY_AVX2
      separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpCubic>(
        output, input, align_corners, {scales_h, scales_w},
        /*antialias=*/false);
    #endif  // CPU_CAPABILITY_AVX2
  }
  else {
    upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpCubic>(
      output, input, align_corners, {scales_h, scales_w});
  }
}

void upsample_bicubic2d_aa_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {

#ifdef CPU_CAPABILITY_AVX2
  if (input.dtype() == at::kByte && input.size(1) <= 4) {
    upsample_avx_bilinear_bicubic_uint8<scale_t, HelperInterpCubic>(
      input, output, align_corners, {scales_h, scales_w},
      /*antialias=*/true);
  } else {
    separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpCubic>(
        output, input, align_corners, {scales_h, scales_w},
        /*antialias=*/true);
  }
#else // CPU_CAPABILITY_AVX2
  separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpCubic>(
      output, input, align_corners, {scales_h, scales_w},
      /*antialias=*/true);
#endif // CPU_CAPABILITY_AVX2
}

template <
    typename scalar_t,
    typename scale_type,
    class F>
void cpu_upsample_genNd_backward_aa(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    bool align_corners,
    const scale_type& scales) {
  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());

  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();
  auto ndim = input_sizes.size();

  // treat nbatch and channels as one dimension
  int64_t channels = input_sizes[0] * input_sizes[1];
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  int64_t output_slice_size = output_depth * output_height * output_width;
  int interp_size = F::interp_size;

  auto loop2d = [&](int64_t begin, int64_t end) {
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[0]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[1]);

    auto input_indexr = [=](int64_t c, int64_t h, int64_t w) {
      return grad_input_data + c * input_height * input_width +
          h * input_width + w;
    };

    const scalar_t support_h = (height_scale >= 1.0)
        ? (interp_size * 0.5) * height_scale
        : interp_size * 0.5;
    const scalar_t support_w = (width_scale >= 1.0)
        ? (interp_size * 0.5) * width_scale
        : interp_size * 0.5;

    const int interp_height = (int)ceilf(support_h) * 2 + 1;
    const int interp_width = (int)ceilf(support_w) * 2 + 1;

    std::vector<scalar_t> wx(interp_width, 0.0);
    std::vector<scalar_t> wy(interp_height, 0.0);

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t xmin, ymin;
    int64_t xsize, ysize;

    typedef scalar_t (*aa_filter_fn_t)(scalar_t);
    aa_filter_fn_t filter_fn = &F::aa_filter;

    for (const auto oh : c10::irange(output_height)) {
      F::_compute_indices_min_size_weights_aa(
          oh,
          input_height,
          height_scale,
          support_h,
          wy.data(),
          interp_height,
          filter_fn,
          ymin,
          ysize);

      for (const auto ow : c10::irange(output_width)) {
        F::_compute_indices_min_size_weights_aa(
            ow,
            input_width,
            width_scale,
            support_w,
            wx.data(),
            interp_width,
            filter_fn,
            xmin,
            xsize);

        for (const auto c : c10::irange(begin, end)) {
          scalar_t grad_output_value =
              grad_output_data[c * output_slice_size + oh * output_width + ow];

          for (const auto y : c10::irange(ysize)) {
            for (const auto x : c10::irange(xsize)) {
              *input_indexr(c, ymin + y, xmin + x) +=
                  wx[x] * wy[y] * grad_output_value;
            }
          }
        }
      }
    }
  };

  if (ndim == 4) {
    // upsample bilinear 2d
    at::parallel_for(
        0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    TORCH_CHECK(false, "Unsupported tensor ndim");
  }

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

void upsample_bilinear2d_aa_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "upsample_bilinear2d_aa_backward_cpu", [&] {
        cpu_upsample_genNd_backward_aa<scalar_t, scale_t, HelperInterpLinear>(
            grad_input, grad_output, align_corners, {scales_h, scales_w});
      });
}

void upsample_bicubic2d_aa_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "upsample_bicubic2d_aa_backward_cpu", [&] {
        cpu_upsample_genNd_backward_aa<scalar_t, scale_t, HelperInterpCubic>(
            grad_input, grad_output, align_corners, {scales_h, scales_w});
      });
}

} // anonymous namespace

REGISTER_DISPATCH(upsample_nearest1d_kernel, &upsample_nearest1d_kernel_impl);
REGISTER_DISPATCH(_upsample_nearest_exact1d_kernel, &_upsample_nearest_exact1d_kernel_impl);
REGISTER_DISPATCH(upsample_nearest2d_kernel, &upsample_nearest2d_kernel_impl);
REGISTER_DISPATCH(_upsample_nearest_exact2d_kernel, &_upsample_nearest_exact2d_kernel_impl);
REGISTER_DISPATCH(upsample_nearest3d_kernel, &upsample_nearest3d_kernel_impl);
REGISTER_DISPATCH(_upsample_nearest_exact3d_kernel, &_upsample_nearest_exact3d_kernel_impl);

REGISTER_DISPATCH(upsample_linear1d_kernel, &upsample_linear1d_kernel_impl);
REGISTER_DISPATCH(upsample_bilinear2d_kernel, &upsample_bilinear2d_kernel_impl);
REGISTER_DISPATCH(_upsample_bilinear2d_aa_kernel, &upsample_bilinear2d_aa_kernel_impl);
REGISTER_DISPATCH(_upsample_bilinear2d_aa_backward_kernel, &upsample_bilinear2d_aa_backward_kernel_impl);
REGISTER_DISPATCH(upsample_trilinear3d_kernel, &upsample_trilinear3d_kernel_impl);

REGISTER_DISPATCH(upsample_bicubic2d_kernel, &upsample_bicubic2d_kernel_impl);
REGISTER_DISPATCH(_upsample_bicubic2d_aa_kernel, &upsample_bicubic2d_aa_kernel_impl);
REGISTER_DISPATCH(_upsample_bicubic2d_aa_backward_kernel, &upsample_bicubic2d_aa_backward_kernel_impl);
} // namespace at::native

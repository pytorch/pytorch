#include <math.h>
#include <vector>
#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/UpSample.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace {

using scale_t = std::vector<c10::optional<double>>;


template<typename scalar_t>
static inline void compute_source_index_and_lambda(
    int64_t& input_index0,
    int64_t& input_index1,
    scalar_t& lambda0,
    scalar_t& lambda1,
    scalar_t ratio,
    int64_t output_index,
    int64_t input_size,
    int64_t output_size,
    bool align_corners) {
  if (output_size == input_size) {
    // scale_factor = 1, simply copy
    input_index0 = output_index;
    input_index1 = output_index;
    lambda0 = static_cast<scalar_t>(1);
    lambda1 = static_cast<scalar_t>(0);
  } else {
    const scalar_t real_input_index = area_pixel_compute_source_index<scalar_t>(
        ratio, output_index, align_corners, /*cubic=*/false);
    input_index0 = static_cast<int64_t>(real_input_index);
    int64_t offset = (input_index0 < input_size - 1) ? 1 : 0;
    input_index1 = input_index0 + offset;
    lambda1 = real_input_index - input_index0;
    lambda0 = static_cast<scalar_t>(1.) - lambda1;
  }
}

// Helper structs and methods for cpu_upsample_linear
//
// Interpolation methods that used below are separable, and as such we can compute the interpolation
// independently per dimension in a recursive way. Please, refer to #10482 for more context.
//
// Linear Interpolation structure to compute output value in n-dimensional case.
// - recursively compute interpolated output for each dimension
// - we rely a lot on compiler's code optimization such that implemented operations
//   can be automatically factorized and vectorized using SSE and AVX2
template <int n, typename scalar_t, typename index_t>
struct InterpLinear {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        index_t i0 = *(index_t*)&data[0][i * strides[0]];
        index_t i1 = *(index_t*)&data[2][i * strides[2]];
        scalar_t w0 = *(scalar_t *)&data[1][i * strides[1]];
        scalar_t w1 = *(scalar_t *)&data[3][i * strides[3]];

        scalar_t t0 = InterpLinear<n - 1, scalar_t, index_t>::eval(src + i0, &data[4], &strides[4], i);
        scalar_t t1 = InterpLinear<n - 1, scalar_t, index_t>::eval(src + i1, &data[4], &strides[4], i);

        return t0 * w0 + t1 * w1;
    }
};

template <typename scalar_t, typename index_t>
struct InterpLinear<1, scalar_t, index_t> {
    static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        index_t i0 = *(index_t*)&data[0][i * strides[0]];
        index_t i1 = *(index_t*)&data[2][i * strides[2]];
        scalar_t w0 = *(scalar_t *)&data[1][i * strides[1]];
        scalar_t w1 = *(scalar_t *)&data[3][i * strides[3]];
        scalar_t t0 = *(scalar_t *)&src[i0];
        scalar_t t1 = *(scalar_t *)&src[i1];
        return t0 * w0 + t1 * w1;
    }
};

template <int n, typename scalar_t, typename index_t>
static inline scalar_t interp_linear(char* src, char** data, const int64_t* strides, int64_t i) {
  return InterpLinear<n, scalar_t, index_t>::eval(src, data, strides, i);
}

static inline bool is_zero_stride(const int64_t* strides) {
  return (strides[0] == 0) && (strides[1] == 0) && (strides[2] == 0) && (strides[3] == 0);
}

template <typename scalar_t, typename index_t>
static inline bool is_contiguous_stride(const int64_t* strides) {
  return (strides[0] == sizeof(index_t)) && (strides[1] == sizeof(scalar_t)) &&
         (strides[2] == sizeof(index_t)) && (strides[3] == sizeof(scalar_t));
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
template <int N, int non_zero_stride_dim, typename scalar_t, typename index_t>
struct CheckAlmostAllZeroStrides {
  static inline bool eval(const int64_t* strides) {
    return (N == non_zero_stride_dim ? is_contiguous_stride<scalar_t, index_t>(strides) : is_zero_stride(strides)) &&
            CheckAlmostAllZeroStrides<N - 1, non_zero_stride_dim, scalar_t, index_t>::eval(&strides[4]);
  }
};

template <int non_zero_stride_dim, typename scalar_t, typename index_t>
struct CheckAlmostAllZeroStrides<0, non_zero_stride_dim, scalar_t, index_t> {
  static inline bool eval(const int64_t* strides) {
    return true;
  }
};

template <int n, int s, typename scalar_t, typename index_t>
static inline bool is_all_zero_stride(const int64_t* strides) {
  return CheckAlmostAllZeroStrides<n, s, scalar_t, index_t>::eval(strides);
}

// Helper method to compute linear interpolation
template <typename scalar_t, typename index_t, int out_ndims>
static inline void basic_loop(char** data, const int64_t* strides, int64_t n) {
  char* dst = data[0];
  char* src = data[1];
  for (int64_t i = 0; i < n; i++) {
    *(scalar_t*)&dst[i * strides[0]] = interp_linear<out_ndims, scalar_t, index_t>(
        src + i * strides[1], &data[2], &strides[2], i);
  }
}

// Linear upsampling computation method using TensorIterator for Nd case.
//
// Single loop function for 1d, 2d and 3d cases.
// For N dimensions, output value up to Di dimension can be computed as
//
// output_i[a] = linear_interp(output_{i+1}[a], w_{i+1}[a], output_{i+1}[a+1], w_{i+1}[a+1])
// with
// output_DN[a] = linear_interp(input_DN[a], w_DN[a], input_DN[a+1], w_DN[a+1])
//
// The recursive call is implemented with InterpLinear struct using template for
// the loop unrolling on compile time.
template <typename scalar_t, int out_ndims>
void cpu_upsample_linear(at::TensorIterator& iter)
{
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    // special-cases to let the compiler apply compile-time input-specific optimizations
    if ((strides[0] == sizeof(scalar_t) && (strides[1] == 0) &&
        is_all_zero_stride<out_ndims, 1, scalar_t, int64_t>(&strides[2]))) {
      // contiguous channels-first case
      basic_loop<scalar_t, int64_t, out_ndims>(data, strides, n);
    } else if ((strides[0] == sizeof(scalar_t) && (strides[1] == sizeof(scalar_t)) &&
               is_all_zero_stride<out_ndims, -1, scalar_t, int64_t>(&strides[2]))) {
      // contiguous channels-last case
      basic_loop<scalar_t, int64_t, out_ndims>(data, strides, n);
    } else {
      // fallback
      basic_loop<scalar_t, int64_t, out_ndims>(data, strides, n);
    }
  };
  iter.for_each(loop);
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

  auto input_data = input.data_ptr<scalar_t>();
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

  using Vec = vec256::Vec256<scalar_t>;
  auto loop2d = [&](int64_t begin, int64_t end) {
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[0]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[1]);

    auto input_indexr = [=](int64_t n, int64_t h, int64_t w) {
      return input_data + n * input_height * input_width * channels +
          h * input_width * channels + w * channels;
    };

    int64_t ih0, ih1, iw0, iw1;
    scalar_t h0lambda, h1lambda, w0lambda, w1lambda;
    for (int64_t n = begin; n < end; n++) {
      for (int64_t oh = 0; oh < output_height; oh++) {
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        for (int64_t ow = 0; ow < output_width; ow++) {
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);

          scalar_t* out = output_data + n * output_slice_size +
              oh * output_width * channels + ow * channels;
          scalar_t* i00 = input_indexr(n, ih0, iw0);
          scalar_t* i01 = input_indexr(n, ih0, iw1);
          scalar_t* i10 = input_indexr(n, ih1, iw0);
          scalar_t* i11 = input_indexr(n, ih1, iw1);

          int64_t size = channels;
          int64_t d = 0;
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            Vec out_vec =
                Vec(h0lambda * w0lambda) * Vec::loadu(i00 + d) + /* h0 * w0 * i00 */
                Vec(h0lambda * w1lambda) * Vec::loadu(i01 + d) + /* h0 * w1 * i01 */
                Vec(h1lambda * w0lambda) * Vec::loadu(i10 + d) + /* h1 * w0 * i10 */
                Vec(h1lambda * w1lambda) * Vec::loadu(i11 + d);  /* h1 * w1 * i11 */
            out_vec.store(out + d);
          }
          for (; d < size; d++) {
            out[d] =
                h0lambda * w0lambda * i00[d] + /* h0 * w0 * i00 */
                h0lambda * w1lambda * i01[d] + /* h0 * w1 * i01 */
                h1lambda * w0lambda * i10[d] + /* h1 * w0 * i10 */
                h1lambda * w1lambda * i11[d];  /* h1 * w1 * i11 */
          }
        }
      }
    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    const scalar_t depth_scale = area_pixel_compute_scale<scalar_t>(
        input_depth, output_depth, align_corners, scales[0]);
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[1]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[2]);

    auto input_indexr = [=](int64_t n, int64_t d, int64_t h, int64_t w) {
      return input_data + n * input_depth * input_height * input_width * channels +
          d * input_height * input_width * channels +
          h * input_width * channels + w * channels;
    };

    int64_t id0, id1, ih0, ih1, iw0, iw1;
    scalar_t d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
    for (int64_t n = begin; n < end; n++) {
      for (int64_t od = 0; od < output_depth; od++) {
        compute_source_index_and_lambda(
            id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
        for (int64_t oh = 0; oh < output_height; oh++) {
          compute_source_index_and_lambda(
              ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
          for (int64_t ow = 0; ow < output_width; ow++) {
            compute_source_index_and_lambda(
                iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);

            scalar_t* out = output_data + n * output_slice_size +
                od * output_height * output_width * channels +
                oh * output_width * channels + ow * channels;
            scalar_t* i000 = input_indexr(n, id0, ih0, iw0);
            scalar_t* i001 = input_indexr(n, id0, ih0, iw1);
            scalar_t* i010 = input_indexr(n, id0, ih1, iw0);
            scalar_t* i011 = input_indexr(n, id0, ih1, iw1);
            scalar_t* i100 = input_indexr(n, id1, ih0, iw0);
            scalar_t* i101 = input_indexr(n, id1, ih0, iw1);
            scalar_t* i110 = input_indexr(n, id1, ih1, iw0);
            scalar_t* i111 = input_indexr(n, id1, ih1, iw1);

            int64_t size = channels;
            int64_t d = 0;
            for (; d < size - (size % Vec::size()); d += Vec::size()) {
              Vec out_vec =
                  Vec(d0lambda * h0lambda * w0lambda) * Vec::loadu(i000 + d) + /* d0 * h0 * w0 * i000 */
                  Vec(d0lambda * h0lambda * w1lambda) * Vec::loadu(i001 + d) + /* d0 * h0 * w1 * i001 */
                  Vec(d0lambda * h1lambda * w0lambda) * Vec::loadu(i010 + d) + /* d0 * h1 * w0 * i010 */
                  Vec(d0lambda * h1lambda * w1lambda) * Vec::loadu(i011 + d) + /* d0 * h1 * w1 * i011 */
                  Vec(d1lambda * h0lambda * w0lambda) * Vec::loadu(i100 + d) + /* d1 * h0 * w0 * i100 */
                  Vec(d1lambda * h0lambda * w1lambda) * Vec::loadu(i101 + d) + /* d1 * h0 * w1 * i101 */
                  Vec(d1lambda * h1lambda * w0lambda) * Vec::loadu(i110 + d) + /* d1 * h1 * w0 * i110 */
                  Vec(d1lambda * h1lambda * w1lambda) * Vec::loadu(i111 + d);  /* d1 * h1 * w1 * i111 */
              out_vec.store(out + d);
            }
            for (; d < size; d++) {
              out[d] =
                  d0lambda * h0lambda * w0lambda * i000[d] + /* d0 * h0 * w0 * i000 */
                  d0lambda * h0lambda * w1lambda * i001[d] + /* d0 * h0 * w1 * i001 */
                  d0lambda * h1lambda * w0lambda * i010[d] + /* d0 * h1 * w0 * i010 */
                  d0lambda * h1lambda * w1lambda * i011[d] + /* d0 * h1 * w1 * i011 */
                  d1lambda * h0lambda * w0lambda * i100[d] + /* d1 * h0 * w0 * i100 */
                  d1lambda * h0lambda * w1lambda * i101[d] + /* d1 * h0 * w1 * i101 */
                  d1lambda * h1lambda * w0lambda * i110[d] + /* d1 * h1 * w0 * i110 */
                  d1lambda * h1lambda * w1lambda * i111[d];  /* d1 * h1 * w1 * i111 */
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

template <typename scalar_t, typename scale_type>
void cpu_upsample_linear_backward(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    bool align_corners,
    const scale_type& scales) {
  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());

  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();
  auto ndim = input_sizes.size();

  // treat nbatch and channels as one dimension
  int64_t channels = input_sizes[0] * input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  int64_t output_slice_size = output_depth * output_height * output_width;

  auto loop1d = [&](int64_t begin, int64_t end) {
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[0]);

    auto input_indexr = [=](int64_t c, int64_t w) {
      return grad_input_data + c * input_width + w;
    };

    int64_t iw0, iw1;
    scalar_t w0lambda, w1lambda;
    for (int64_t c = begin; c < end; c++){
      for (int64_t ow = 0; ow < output_width; ow++) {
        compute_source_index_and_lambda(
            iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
        scalar_t grad_output_value = grad_output_data[c * output_slice_size + ow];
        *input_indexr(c, iw0) += w0lambda * grad_output_value; /* i0 */
        *input_indexr(c, iw1) += w1lambda * grad_output_value; /* i1*/
      }
    }
  };

  auto loop2d = [&](int64_t begin, int64_t end) {
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[0]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[1]);

    auto input_indexr = [=](int64_t c, int64_t h, int64_t w){
      return grad_input_data + c * input_height * input_width + h * input_width + w;
    };

    int64_t ih0, ih1, iw0, iw1;
    scalar_t h0lambda, h1lambda, w0lambda, w1lambda;
    for (int64_t c = begin; c < end; c++) {
      for (int64_t oh = 0; oh < output_height; oh++) {
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        for (int64_t ow = 0; ow < output_width; ow++) {
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
          scalar_t grad_output_value = grad_output_data[c * output_slice_size + oh * output_width + ow];
          *input_indexr(c, ih0, iw0) += h0lambda * w0lambda * grad_output_value; /* i00 */
          *input_indexr(c, ih0, iw1) += h0lambda * w1lambda * grad_output_value; /* i01 */
          *input_indexr(c, ih1, iw0) += h1lambda * w0lambda * grad_output_value; /* i10 */
          *input_indexr(c, ih1, iw1) += h1lambda * w1lambda * grad_output_value; /* i11 */
        }
      }
    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    const scalar_t depth_scale = area_pixel_compute_scale<scalar_t>(
        input_depth, output_depth, align_corners, scales[0]);
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[1]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[2]);

    auto input_indexr = [=](int64_t c, int64_t d, int64_t h, int64_t w) {
      return grad_input_data + c * input_depth * input_height * input_width +
          d * input_height * input_width + h * input_width + w;
    };

    int64_t id0, id1, ih0, ih1, iw0, iw1;
    scalar_t d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
    for (int64_t c = begin; c < end; c++) {
      for (int64_t od = 0; od < output_depth; od++) {
        compute_source_index_and_lambda(
            id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
        for (int64_t oh = 0; oh < output_height; oh++) {
          compute_source_index_and_lambda(
              ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
          for (int64_t ow = 0; ow < output_width; ow++) {
            compute_source_index_and_lambda(
                iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
            scalar_t grad_output_value = grad_output_data[c * output_slice_size +
                od *  output_height * output_width + oh * output_width + ow];
            *input_indexr(c, id0, ih0, iw0) += d0lambda * h0lambda * w0lambda * grad_output_value; /* i000 */
            *input_indexr(c, id0, ih0, iw1) += d0lambda * h0lambda * w1lambda * grad_output_value; /* i001 */
            *input_indexr(c, id0, ih1, iw0) += d0lambda * h1lambda * w0lambda * grad_output_value; /* i010 */
            *input_indexr(c, id0, ih1, iw1) += d0lambda * h1lambda * w1lambda * grad_output_value; /* i011 */
            *input_indexr(c, id1, ih0, iw0) += d1lambda * h0lambda * w0lambda * grad_output_value; /* i100 */
            *input_indexr(c, id1, ih0, iw1) += d1lambda * h0lambda * w1lambda * grad_output_value; /* i101 */
            *input_indexr(c, id1, ih1, iw0) += d1lambda * h1lambda * w0lambda * grad_output_value; /* i110 */
            *input_indexr(c, id1, ih1, iw1) += d1lambda * h1lambda * w1lambda * grad_output_value; /* i111 */
          }
        }
      }
    }
  };

  if (ndim == 3) {
    // upsample linear 1d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 2, loop1d);
  } else if (ndim == 4) {
    // upsample bilinear 2d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    // upsample trilinear 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 8, loop3d);
  }

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}


// Method to compute indices and weights for each interpolated dimension
// indices_weights = {
//      {indices_0, weights_0, indices_1, weights_1},  // dim -n
//      {indices_0, weights_0, indices_1, weights_1},  // dim -(n-1)
//      ...
//      {indices_0, weights_0, indices_1, weights_1},  // dim -1
// }
// Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to
// fit input/output tensors.
// Indices are already containing the strides to optimize the computations
//
template<typename scalar_t>
std::vector<Tensor> compute_indices_weights_linear(
  int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims, int64_t reshape_dim,
  bool align_corners, const c10::optional<double> opt_scale
) {

  scalar_t scale = area_pixel_compute_scale<scalar_t>(input_size, output_size, align_corners, opt_scale);

  std::vector<Tensor> output;
  auto new_shape = std::vector<int64_t>(ndims, 1);
  new_shape[reshape_dim] = output_size;

  output.emplace_back(empty(new_shape, CPU(at::kLong)));
  output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<scalar_t>())));
  output.emplace_back(empty(new_shape, CPU(at::kLong)));
  output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<scalar_t>())));

  auto input_index0_ptr = output[0].data_ptr<int64_t>();
  auto lambda0_ptr = output[1].data_ptr<scalar_t>();
  auto input_index1_ptr = output[2].data_ptr<int64_t>();
  auto lambda1_ptr = output[3].data_ptr<scalar_t>();

  for (int64_t i=0; i<output_size; i++) {

    compute_source_index_and_lambda<scalar_t>(
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
  return output;
}

// Upsampling linear interpolation kernel for N-d case.
// Input is assumed to be like NCHW, NCL, NCKHW - interpolated spatial dimension
// are those from the end up to batch size N and number of channels C.
//
// Internally, it uses TensorIterator to optimize the computations.
// - out_ndims is the number of interpolated dims: 1, 2, 3
// - scale_type is template type for scales, typically c10::optional<double>
template <int out_ndims, typename scale_type>
void upsample_linearNd_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    const scale_type& scales) {

  // We apply a similar logic as in advanced indexing implementation
  // - output spatial dimensions are different from input ones and
  //   we have to restride the input to have strides equal zeron on
  //   spatial dimensions
  auto shape = input.sizes().vec();
  auto strides = input.strides().vec();
  auto oshape = output.sizes();

  for (int i=0; i<out_ndims; i++) {
    shape[i + 2] = oshape[i + 2];
    strides[i + 2] = 0;
  }
  auto restrided_input = input.as_strided(shape, strides);

  std::vector<std::vector<Tensor>> indices_weights;
  AT_DISPATCH_FLOATING_TYPES(
    input.scalar_type(), "compute_indices_weights_linear", [&] {
      auto es = input.element_size();
      for (int i=0; i<out_ndims; i++) {
        indices_weights.emplace_back(
          compute_indices_weights_linear<scalar_t>(
            input.size(i + 2), oshape[i + 2], input.stride(i + 2) * es, input.dim(), i + 2, align_corners, scales[i])
        );
      }
    }
  );

  TensorIteratorConfig config;
  config.check_all_same_dtype(false)
    .declare_static_dtype_and_device(input.scalar_type(), input.device())
    .add_output(output)
    .add_input(restrided_input);

  for (auto iter=indices_weights.begin(); iter!=indices_weights.end(); iter++) {
    for (auto& tensor : *iter) {
      config.add_input(tensor);
    }
  }

  auto iter = config.build();

  AT_DISPATCH_FLOATING_TYPES(
      iter.dtype(), "upsample_linearNd", [&] {
      cpu_upsample_linear<scalar_t, out_ndims>(iter);
  });

}

void upsample_linear1d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_w) {
  upsample_linearNd_kernel_impl<1, scale_t>(
    output, input, align_corners, {scales_w});
}

void upsample_bilinear2d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_bilinear2d_channels_last", [&] {
      cpu_upsample_linear_channels_last<scalar_t, scale_t>(output, input, align_corners, {scales_h, scales_w});
    });
  } else {
    upsample_linearNd_kernel_impl<2, scale_t>(output, input, align_corners, {scales_h, scales_w});
  }
}

void upsample_trilinear3d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (input.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_trilinear3d_channels_last", [&] {
      cpu_upsample_linear_channels_last<scalar_t, scale_t>(output, input, align_corners, {scales_d, scales_h, scales_w});
    });
  } else {
    upsample_linearNd_kernel_impl<3, scale_t>(output, input, align_corners, {scales_d, scales_h, scales_w});
  }
}

void upsample_linear1d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_linear1d_backward", [&] {
    cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_w});
  });
}

void upsample_bilinear2d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_bilinear2d_backward", [&] {
    cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_h, scales_w});
  });
}

void upsample_trilinear3d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_trilinear3d_backward", [&] {
    cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_d, scales_h, scales_w});
  });
}

} // anonymous namespace

REGISTER_DISPATCH(upsample_linear1d_kernel, &upsample_linear1d_kernel_impl);
REGISTER_DISPATCH(upsample_bilinear2d_kernel, &upsample_bilinear2d_kernel_impl);
REGISTER_DISPATCH(upsample_trilinear3d_kernel, &upsample_trilinear3d_kernel_impl);
REGISTER_DISPATCH(upsample_linear1d_backward_kernel, &upsample_linear1d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_bilinear2d_backward_kernel, &upsample_bilinear2d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_trilinear3d_backward_kernel, &upsample_trilinear3d_backward_kernel_impl);

} // namespace native
} // namespace at

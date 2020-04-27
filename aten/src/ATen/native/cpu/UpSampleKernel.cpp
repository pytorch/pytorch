#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/UpSample.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at {
namespace native {
namespace {

template <typename T>
inline T data_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T &x, const T &X, Args &&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T &x, const T &X, Args &&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}

static inline int64_t nearest_idx(
    int64_t output_index,
    int64_t input_size,
    int64_t output_size,
    c10::optional<double> scales) {
  if (output_size == input_size) {
    // scale_factor = 1, simply copy
    return output_index;
  } else if (output_size == 2 * input_size) {
    // scale_factor = 2, shift input index
    return output_index >> 1;
  } else {
    float scale = compute_scales_value<float>(scales, input_size, output_size);
    return nearest_neighbor_compute_source_index(scale, output_index, input_size);
  }
}

template <typename scalar_t, typename scale_type>
void cpu_upsample_nearest(
    Tensor& output_,
    const Tensor& input_,
    const scale_type& scales) {
  TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
              " for `output` but got dtype ", output_.dtype());
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto input_sizes = input.sizes().vec();
  auto output_sizes = output.sizes().vec();
  auto ndim = input_sizes.size();
  auto numel = output.numel();

  // treat nbatch and channels as one dimension
  int64_t channels = input_sizes[0] * input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  auto loop1d = [&](int64_t start, int64_t end) {
    int64_t c = 0;
    int64_t ow = 0;
    data_index_init(start, c, channels, ow, output_width);
    for (int64_t i = start; i < end; i++) {
      int64_t iw = nearest_idx(ow, input_width, output_width, scales[0]);
      output_data[i] = input_data[c * input_width + iw];
      data_index_step(c, channels, ow, output_width);
    }
  };

  auto loop2d = [&](int64_t start, int64_t end) {
    int64_t c = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(start, c, channels, oh, output_height, ow, output_width);

    for (int64_t i = start; i < end; i++) {
      int64_t ih = nearest_idx(oh, input_height, output_height, scales[0]);
      int64_t iw = nearest_idx(ow, input_width, output_width, scales[1]);
      output_data[i] = input_data[c * input_height * input_width + ih * input_width + iw];
      data_index_step(c, channels, oh, output_height, ow, output_width);
    }
  };

  auto loop3d = [&](int64_t start, int64_t end) {
    int64_t c = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(start, c, channels, od, output_depth, oh, output_height, ow, output_width);

    for (int64_t i = start; i < end; i++) {
      int64_t id = nearest_idx(od, input_depth, output_depth, scales[0]);
      int64_t ih = nearest_idx(oh, input_height, output_height, scales[1]);
      int64_t iw = nearest_idx(ow, input_width, output_width, scales[2]);
      int64_t j = c * input_depth * input_height * input_width +
                  id * input_height * input_width + ih * input_width + iw;
      output_data[i] = input_data[j];
      data_index_step(c, channels, od, output_depth, oh, output_height, ow, output_width);
    }
  };

  if (ndim == 3) {
    // upsample nearest 1d
    at::parallel_for(0, numel, at::internal::GRAIN_SIZE, loop1d);
  } else if (ndim == 4) {
    // upsample nearest 2d
    at::parallel_for(0, numel, at::internal::GRAIN_SIZE, loop2d);
  } else {
    // upsample nearest 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, numel, at::internal::GRAIN_SIZE, loop3d);
  }

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

template <typename scalar_t, typename scale_type>
void cpu_upsample_nearest_channels_last(
    Tensor& output_,
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
  int64_t numel = output.numel();

  using Vec = vec256::Vec256<scalar_t>;
  auto loop2d = [&](int64_t start, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(start, n, num_batches, oh, output_height, ow, output_width);

    for (int64_t i = start; i < end; i++) {
      int64_t ih = nearest_idx(oh, input_height, output_height, scales[0]);
      int64_t iw = nearest_idx(ow, input_width, output_width, scales[1]);
      int64_t j = n * (input_height * input_width * channels)
          + ih * (input_width * channels) + iw * channels;
      vec256::map(
          [](Vec x) { return x; },
          &output_data[i],
          &input_data[j],
          channels);
      data_index_step(n, num_batches, oh, output_height, ow, output_width);
    }
  };

  auto loop3d = [&](int64_t start, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(start, n, num_batches, oh, output_height, ow, output_width);
    data_index_init(start, n, num_batches, od, output_depth, oh, output_height, ow, output_width);

    for (int64_t i = start; i < end; i++) {
      int64_t id = nearest_idx(od, input_depth, output_depth, scales[0]);
      int64_t ih = nearest_idx(oh, input_height, output_height, scales[1]);
      int64_t iw = nearest_idx(ow, input_width, output_width, scales[2]);
      int64_t j = n * (input_depth * input_height * input_width * channels)
          + id * (input_height * input_width * channels)
          + ih * (input_width * channels) + iw * channels;
      vec256::map(
          [](Vec x) { return x; },
          &output_data[i],
          &input_data[j],
          channels);
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

template <typename scalar_t, typename scale_type>
void cpu_upsample_nearest_backward(
    Tensor& grad_input_,
    const Tensor& grad_output_,
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
  int64_t input_slice_size = input_depth * input_height * input_width;

  auto loop1d = [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++){
      for (int64_t ow = 0; ow < output_width; ow++) {
        int64_t iw = nearest_idx(ow, input_width, output_width, scales[0]);
        int64_t output_offset = c * output_slice_size + ow;
        int64_t input_offset = c * input_slice_size + iw;
        grad_input_data[input_offset] += grad_output_data[output_offset];
      }
    }
  };

  auto loop2d = [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      for (int64_t oh = 0; oh < output_height; oh++) {
        int64_t ih = nearest_idx(oh, input_height, output_height, scales[0]);
        for (int64_t ow = 0; ow < output_width; ow++) {
          int64_t iw = nearest_idx(ow, input_width, output_width, scales[1]);
          int64_t output_offset = c * output_slice_size + oh * output_width + ow;
          int64_t input_offset = c * input_slice_size + ih * input_width + iw;
          grad_input_data[input_offset] += grad_output_data[output_offset];
        }
      }
    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      for (int64_t od = 0; od < output_depth; od++) {
        int64_t id = nearest_idx(od, input_depth, output_depth, scales[0]);
        for (int64_t oh = 0; oh < output_height; oh++) {
          int64_t ih = nearest_idx(oh, input_height, output_height, scales[1]);
          for (int64_t ow = 0; ow < output_width; ow++) {
            int64_t iw = nearest_idx(ow, input_width, output_width, scales[2]);
            int64_t output_offset = c * output_slice_size +
                od *  output_height * output_width + oh * output_width + ow;
            int64_t input_offset = c * input_slice_size +
                id * input_height * input_width + ih * input_width + iw;
            grad_input_data[input_offset] += grad_output_data[output_offset];
          }
        }
      }
    }
  };

  if (ndim == 3) {
    // upsample nearest 1d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size, loop1d);
  } else if (ndim == 4) {
    // upsample nearest 2d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size , loop2d);
  } else {
    // upsample nearest 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size, loop3d);
  }

  // Should this actually check for is_contiguous(input.suggest_memory_format())
  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

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

template <typename scalar_t, typename scale_type>
void cpu_upsample_linear(
    Tensor& output_,
    const Tensor& input_,
    bool align_corners,
    const scale_type& scales) {
  TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
              " for `output` but got dtype ", output_.dtype());
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto input_sizes = input.sizes().vec();
  auto output_sizes = output.sizes().vec();
  auto ndim = input_sizes.size();
  auto numel = output.numel();

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
      return input_data[c * input_width + w];
    };

    int64_t iw0, iw1;
    scalar_t w0lambda, w1lambda;
    for (int64_t c = begin; c < end; c++) {
      for (int64_t ow = 0; ow < output_width; ow++) {
        compute_source_index_and_lambda(
            iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
        int64_t output_offset = c * output_slice_size + ow;
        output_data[output_offset] =
            w0lambda * input_indexr(c, iw0) + /* w0 * i0 */
            w1lambda * input_indexr(c, iw1);  /* w1 * i1 */
      }
    }
  };

  auto loop2d = [&](int64_t begin, int64_t end) {
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[0]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[1]);

    auto input_indexr = [=](int64_t c, int64_t h, int64_t w) {
      return input_data[c * input_height * input_width + h * input_width + w];
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
          int64_t output_offset = c * output_slice_size + oh * output_width + ow;
          output_data[output_offset] =
              h0lambda * w0lambda * input_indexr(c, ih0, iw0) + /* h0 * w0 * i00 */
              h0lambda * w1lambda * input_indexr(c, ih0, iw1) + /* h0 * w1 * i01 */
              h1lambda * w0lambda * input_indexr(c, ih1, iw0) + /* h1 * w0 * i10 */
              h1lambda * w1lambda * input_indexr(c, ih1, iw1);  /* h1 * w1 * i11 */
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
      return input_data[c * input_depth * input_height * input_width +
          d * input_height * input_width + h * input_width + w];
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
            int64_t output_offset = c * output_slice_size +
                od * output_height * output_width + oh * output_width + ow;
            output_data[output_offset] =
                d0lambda * h0lambda * w0lambda * input_indexr(c, id0, ih0, iw0) + /* d0 * h0 * w0 * i000 */
                d0lambda * h0lambda * w1lambda * input_indexr(c, id0, ih0, iw1) + /* d0 * h0 * w1 * i001 */
                d0lambda * h1lambda * w0lambda * input_indexr(c, id0, ih1, iw0) + /* d0 * h1 * w0 * i010 */
                d0lambda * h1lambda * w1lambda * input_indexr(c, id0, ih1, iw1) + /* d0 * h1 * w1 * i011 */
                d1lambda * h0lambda * w0lambda * input_indexr(c, id1, ih0, iw0) + /* d1 * h0 * w0 * i100 */
                d1lambda * h0lambda * w1lambda * input_indexr(c, id1, ih0, iw1) + /* d1 * h0 * w1 * i101 */
                d1lambda * h1lambda * w0lambda * input_indexr(c, id1, ih1, iw0) + /* d1 * h1 * w0 * i110 */
                d1lambda * h1lambda * w1lambda * input_indexr(c, id1, ih1, iw1);  /* d1 * h1 * w1 * i111 */
          }
        }
      }
    }
  };

  // compared to "nearest" mode, lower the grain size:
  // "linear", "bilinear", "trilinear" mode are more computational expensive
  if (ndim == 3) {
    // upsample linear 1d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 2, loop1d);
  } else if (ndim == 4){
    // upsample bilinear 2d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    // upsample trilinear 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 8, loop3d);
  }

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

template <typename scalar_t, typename scale_type>
void cpu_upsample_linear_backward(
    Tensor& grad_input_,
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

using scale_t = std::vector<c10::optional<double>>;
void upsample_nearest1d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Byte, input.scalar_type(), "upsample_nearest1d", [&] {
    cpu_upsample_nearest<scalar_t, scale_t>(output, input, {scales_w});
  });
}

void upsample_nearest2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Byte, input.scalar_type(), "upsample_nearest2d_channels_last", [&] {
      cpu_upsample_nearest_channels_last<scalar_t, scale_t>(output, input, {scales_h, scales_w});
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Byte, input.scalar_type(), "upsample_nearest2d", [&] {
      cpu_upsample_nearest<scalar_t, scale_t>(output, input, {scales_h, scales_w});
    });
  }
}

void upsample_nearest3d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (input.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Byte, input.scalar_type(), "upsample_nearest3d_channels_last", [&] {
      cpu_upsample_nearest_channels_last<scalar_t, scale_t>(output, input, {scales_d, scales_h, scales_w});
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Byte, input.scalar_type(), "upsample_nearest3d", [&] {
      cpu_upsample_nearest<scalar_t, scale_t>(output, input, {scales_d, scales_h, scales_w});
    });
  }
}

void upsample_nearest1d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "upsample_nearest1d_backward", [&] {
    cpu_upsample_nearest_backward<scalar_t, scale_t>(grad_input, grad_output, {scales_w});
  });
}

void upsample_nearest2d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "upsample_nearest2d_backward", [&] {
    cpu_upsample_nearest_backward<scalar_t, scale_t>(grad_input, grad_output, {scales_h, scales_w});
  });
}

void upsample_nearest3d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "upsample_nearest3d_backward", [&] {
    cpu_upsample_nearest_backward<scalar_t, scale_t>(grad_input, grad_output, {scales_d, scales_h, scales_w});
  });
}

void upsample_linear1d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_linear1d", [&] {
    cpu_upsample_linear<scalar_t, scale_t>(output, input, align_corners, {scales_w});
  });
}

void upsample_bilinear2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_bilinear2d", [&] {
    cpu_upsample_linear<scalar_t, scale_t>(output, input, align_corners, {scales_h, scales_w});
  });
}

void upsample_trilinear3d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_trilinear3d", [&] {
    cpu_upsample_linear<scalar_t, scale_t>(output, input, align_corners, {scales_d, scales_h, scales_w});
  });
}

void upsample_linear1d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "upsample_linear1d_backward", [&] {
    cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_w});
  });
}

void upsample_bilinear2d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "upsample_bilinear2d_backward", [&] {
    cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_h, scales_w});
  });
}

void upsample_trilinear3d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "upsample_trilinear3d_backward", [&] {
    cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_d, scales_h, scales_w});
  });
}

} // anonymous namespace

REGISTER_DISPATCH(upsample_nearest1d_kernel, &upsample_nearest1d_kernel_impl);
REGISTER_DISPATCH(upsample_nearest2d_kernel, &upsample_nearest2d_kernel_impl);
REGISTER_DISPATCH(upsample_nearest3d_kernel, &upsample_nearest3d_kernel_impl);
REGISTER_DISPATCH(upsample_nearest1d_backward_kernel, &upsample_nearest1d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_nearest2d_backward_kernel, &upsample_nearest2d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_nearest3d_backward_kernel, &upsample_nearest3d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_linear1d_kernel, &upsample_linear1d_kernel_impl);
REGISTER_DISPATCH(upsample_bilinear2d_kernel, &upsample_bilinear2d_kernel_impl);
REGISTER_DISPATCH(upsample_trilinear3d_kernel, &upsample_trilinear3d_kernel_impl);
REGISTER_DISPATCH(upsample_linear1d_backward_kernel, &upsample_linear1d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_bilinear2d_backward_kernel, &upsample_bilinear2d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_trilinear3d_backward_kernel, &upsample_trilinear3d_backward_kernel_impl);

} // namespace native
} // namespace at

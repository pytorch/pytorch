#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/UpSample.h>
#include <ATen/Parallel.h>

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
    double scales_1) {
  if (output_size == input_size) {
    // simply copy
    return output_index;
  } else if (output_size == 2 * input_size && scales_1 < 0.) {
    // scale_factor = 2
    return output_index >> 1;
  } else {
    float scale = compute_scales_value<float>(scales_1, input_size, output_size);
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
    at::parallel_for(0, numel, at::internal::GRAIN_SIZE, loop3d);
  }

  if (!output_.is_contiguous()) {
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

  scale_type scales_{scales.begin(), scales.end()};
  while (scales_.size() < 3) {
    scales_.push_back(-1);
  }

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

  auto loop = [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      for (int64_t od = 0; od < output_depth; od++) {
        int64_t id = nearest_idx(od, input_depth, output_depth, scales_[0]);
        for (int64_t oh = 0; oh < output_height; oh++) {
          int64_t ih = nearest_idx(oh, input_height, output_height, scales_[1]);
          for (int64_t ow = 0; ow < output_width; ow++) {
            int64_t iw = nearest_idx(ow, input_width, output_width, scales_[2]);
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

  at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size, loop);

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

using scale_t = std::vector<double>;
void upsample_nearest1d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    double scales_1) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_nearest1d", [&] {
    cpu_upsample_nearest<scalar_t, scale_t>(output, input, {scales_1});
  });
}

void upsample_nearest2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    double scales_1,
    double scales_2) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_nearest2d", [&] {
    cpu_upsample_nearest<scalar_t, scale_t>(output, input, {scales_1, scales_2});
  });
}

void upsample_nearest3d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    double scales_1,
    double scales_2,
    double scales_3) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_nearest3d", [&] {
    cpu_upsample_nearest<scalar_t, scale_t>(output, input, {scales_1, scales_2, scales_3});
  });
}

void upsample_nearest1d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    double scales_1) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "upsample_nearest1d_backward", [&] {
    cpu_upsample_nearest_backward<scalar_t, scale_t>(grad_input, grad_output, {scales_1});
  });
}

void upsample_nearest2d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    double scales_1,
    double scales_2) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "upsample_nearest2d_backward", [&] {
    cpu_upsample_nearest_backward<scalar_t, scale_t>(grad_input, grad_output, {scales_1, scales_2});
  });
}

void upsample_nearest3d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    double scales_1,
    double scales_2,
    double scales_3) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "upsample_nearest3d_backward", [&] {
    cpu_upsample_nearest_backward<scalar_t, scale_t>(grad_input, grad_output, {scales_1, scales_2, scales_3});
  });
}


} // anonymous namespace

REGISTER_DISPATCH(upsample_nearest1d_kernel, &upsample_nearest1d_kernel_impl);
REGISTER_DISPATCH(upsample_nearest2d_kernel, &upsample_nearest2d_kernel_impl);
REGISTER_DISPATCH(upsample_nearest3d_kernel, &upsample_nearest3d_kernel_impl);
REGISTER_DISPATCH(upsample_nearest1d_backward_kernel, &upsample_nearest1d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_nearest2d_backward_kernel, &upsample_nearest2d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_nearest3d_backward_kernel, &upsample_nearest3d_backward_kernel_impl);

} // namespace native
} // namespace at

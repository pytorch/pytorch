#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>

namespace at { namespace native {

namespace {

template <typename scalar_t, bool is_3d>
void cpu_adaptive_avg_pool(
    Tensor& output_,
    const Tensor& input_,
    IntArrayRef output_size) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // treat batch size and channels as one dimension
  //
  // AdaptiveAvgPool2d:
  //   ndim == 3: CHW
  //   ndim == 4: NCHW
  //
  // AdaptiveAvgPool3d:
  //   ndim == 4: CDHW
  //   ndim == 5: NCDHW
  //
  int64_t ndim = input.ndimension();
  int64_t channels;
  if (is_3d) {
    channels = ndim == 4 ? input.size(0) : input.size(0) * input.size(1);
  } else {
    channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
  }
  int64_t input_depth = is_3d ? input.size(-3) : 1;
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_depth = is_3d ? output_size[0] : 1;
  int64_t k = output_size.size();
  int64_t output_height = output_size[k - 2];
  int64_t output_width = output_size[k - 1];

  // pre calculate input indices
  std::vector<PreCalc> pre_calc(
      output_depth * output_height * output_width);

  pre_calc_for_adaptive_pool(
      output_depth, output_height, output_width,
      input_depth, input_height, input_width,
      pre_calc);

  at::parallel_for(0, channels, 0,  [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      scalar_t* input_ptr = input_data + c * input_depth * input_height * input_width;
      scalar_t* output_ptr = output_data + c * output_depth * output_height * output_width;

      for (int64_t index = 0; index < output_depth * output_height * output_width; index++) {
        PreCalc pc = pre_calc[index];
        int64_t k = (pc.id1 - pc.id0) * (pc.ih1 - pc.ih0) * (pc.iw1 - pc.iw0);

        // compute local average
        scalar_t sum = 0;
        for (int64_t id = pc.id0; id < pc.id1; id++) {
          for (int64_t ih = pc.ih0; ih < pc.ih1; ih++) {
            for (int64_t iw = pc.iw0; iw < pc.iw1; iw++) {
              sum += input_ptr[id * input_height * input_width + ih * input_width + iw];
            }
          }
        }
        output_ptr[index] = sum / k;
      }
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

template <typename scalar_t, bool is_3d>
void cpu_adaptive_avg_pool_channels_last(
    Tensor& output_,
    const Tensor& input_,
    IntArrayRef output_size) {
  auto memory_format = is_3d ? at::MemoryFormat::ChannelsLast3d
                             : at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // AdaptiveAvgPool2d: NHWC
  // AdaptiveAvgPool3d: NDHWC
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = is_3d ? input.size(2) : 1;
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_depth = is_3d ? output_size[0] : 1;
  int64_t k = output_size.size();
  int64_t output_height = output_size[k - 2];
  int64_t output_width = output_size[k - 1];

  using Vec = vec::Vectorized<scalar_t>;
  // parallel on dim N, {D}, H, W
  at::parallel_for(0, nbatch * output_depth * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);

    for (int64_t i = begin; i < end; i++) {
      int64_t id0 = start_index(od, output_depth, input_depth);
      int64_t id1 = end_index(od, output_depth, input_depth);
      int64_t kd = id1 - id0;

      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);
      int64_t kh = ih1 - ih0;

      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);
      int64_t kw = iw1 - iw0;

      scalar_t* out = output_data + i * channels;
      int64_t size = channels;

      // Note: For oridinary usage scenario, each out lane should
      //   fit in L1 cache; otherwise consider block dim C.
      // Pass I: zero the out lane
      int64_t d1 = 0;
      for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
        Vec out_vec = Vec(scalar_t(0));
        out_vec.store(out + d1);
      }
      for (; d1 < size; d1++) {
        out[d1] = scalar_t(0);
      }
      // Pass II: compute local sum
      for (int64_t id = id0; id < id1; id++) {
        for (int64_t ih = ih0; ih < ih1; ih++) {
          for (int64_t iw = iw0; iw < iw1; iw++) {
            scalar_t* in = input_data + (n * input_depth * input_height * input_width +
                id * input_height * input_width + ih * input_width + iw) * channels;

            int64_t d2 = 0;
            for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
              Vec out_vec = Vec::loadu(out + d2) + Vec::loadu(in + d2);
              out_vec.store(out + d2);
            }
            for (; d2 < size; d2++) {
              out[d2] += in[d2];
            }
          }
        }
      }
      // Pass III: compute local average
      int64_t d3 = 0;
      for (; d3 < size - (size % Vec::size()); d3 += Vec::size()) {
        Vec out_vec = Vec::loadu(out + d3) / Vec(scalar_t(kd * kh * kw));
        out_vec.store(out + d3);
      }
      for (; d3 < size; d3++) {
        out[d3] = out[d3] / kd / kh / kw;
      }

      // move on to next output index
      data_index_step(n, nbatch, od, output_depth, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t, bool is_3d>
void cpu_adaptive_avg_pool_backward(
    Tensor& grad_input_,
    const Tensor& grad_output_) {
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();

  // treat batch size and channels as one dimension
  //
  // AdaptiveAvgPool2d:
  //   ndim == 3: CHW
  //   ndim == 4: NCHW
  //
  // AdaptiveAvgPool3d:
  //   ndim == 4: CDHW
  //   ndim == 5: NCDHW
  //
  int64_t ndim = grad_output.ndimension();
  int64_t channels;
  if (is_3d) {
    channels = ndim == 4 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  } else {
    channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  }
  int64_t input_depth = is_3d ? grad_input.size(-3) : 1;
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_depth = is_3d ? grad_output.size(-3) : 1;
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // pre calculate input indices
  std::vector<PreCalc> pre_calc(
      output_depth * output_height * output_width);

  pre_calc_for_adaptive_pool(
      output_depth, output_height, output_width,
      input_depth, input_height, input_width,
      pre_calc);

  // parallel on dim of N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      scalar_t* grad_input_ptr = grad_input_data + c * input_depth * input_height * input_width;
      scalar_t* grad_output_ptr = grad_output_data + c * output_depth * output_height * output_width;

      for (int64_t index = 0; index < output_depth * output_height * output_width; index++) {
        PreCalc pc = pre_calc[index];
        int64_t k = (pc.id1 - pc.id0) * (pc.ih1 - pc.ih0) * (pc.iw1 - pc.iw0);

        scalar_t grad_delta = grad_output_ptr[index] / k;
        for (int64_t id = pc.id0; id < pc.id1; id++) {
          for (int64_t ih = pc.ih0; ih < pc.ih1; ih++) {
            for (int64_t iw = pc.iw0; iw < pc.iw1; iw++) {
              int64_t input_offset = id * input_height * input_width + ih * input_width + iw;
              grad_input_ptr[input_offset] += grad_delta;
            }
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t, bool is_3d>
void cpu_adaptive_avg_pool_backward_channels_last(
    Tensor& grad_input_,
    const Tensor& grad_output_) {
  auto memory_format = is_3d ? at::MemoryFormat::ChannelsLast3d
                             : at::MemoryFormat::ChannelsLast;
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);

  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto grad_output_data = grad_output.data_ptr<scalar_t>();

  // AdaptiveAvgPool2d: NHWC
  // AdaptiveAvgPool3d: NDHWC
  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_depth = is_3d ? grad_input.size(2) : 1;
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_depth = is_3d ? grad_output.size(2) : 1;
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  using Vec = vec::Vectorized<scalar_t>;
  // parallel on dim N
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; n++) {
      scalar_t* grad_input_ptr = grad_input_data + n * input_depth * input_height * input_width * channels;
      scalar_t* grad_output_ptr = grad_output_data + n * output_depth * output_height * output_width * channels;

      for (int64_t od = 0; od < output_depth; od++) {
        int64_t id0 = start_index(od, output_depth, input_depth);
        int64_t id1 = end_index(od, output_depth, input_depth);
        int64_t kd = id1 - id0;

        for (int64_t oh = 0; oh < output_height; oh++) {
          int64_t ih0 = start_index(oh, output_height, input_height);
          int64_t ih1 = end_index(oh, output_height, input_height);
          int64_t kh = ih1 - ih0;

          for (int64_t ow = 0; ow < output_width; ow++) {
            int64_t iw0 = start_index(ow, output_width, input_width);
            int64_t iw1 = end_index(ow, output_width, input_width);
            int64_t kw = iw1 - iw0;

            scalar_t* gout = grad_output_ptr + (od * output_height * output_width + oh * output_width + ow) * channels;
            int64_t size = channels;
            for (int64_t id = id0; id < id1; id++) {
              for (int64_t ih = ih0; ih < ih1; ih++) {
                for (int64_t iw = iw0; iw < iw1; iw++) {
                  scalar_t* gin = grad_input_ptr + (id * input_height * input_width + ih * input_width + iw) * channels;

                  int64_t d = 0;
                  for (; d < size - (size % Vec::size()); d += Vec::size()) {
                    Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d) / Vec(scalar_t(kd * kh * kw));
                    gin_vec.store(gin + d);
                  }
                  for (; d < size; d++) {
                    gin[d] += gout[d] / kd / kh / kw;
                  }
                }
              }
            }
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous(memory_format)) {
    grad_input_.copy_(grad_input);
  }
}

void adaptive_avg_pool2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_avg_pool2d", [&] {
        cpu_adaptive_avg_pool<scalar_t, /* is_3d */ false>(output, input, output_size);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_avg_pool2d_channels_last", [&]{
        cpu_adaptive_avg_pool_channels_last<scalar_t, /* is_3d */ false>(output, input, output_size);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void adapative_avg_pool2d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "adaptive_avg_pool2d_backward", [&] {
        cpu_adaptive_avg_pool_backward<scalar_t, /* is_3d */ false>(grad_input, grad_output);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "adaptive_avg_pool2d_backward_channels_last", [&]{
        cpu_adaptive_avg_pool_backward_channels_last<scalar_t, /* is_3d */ false>(grad_input, grad_output);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void adaptive_avg_pool3d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_avg_pool3d", [&] {
        cpu_adaptive_avg_pool<scalar_t, /* is_3d */ true>(output, input, output_size);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_avg_pool3d_channels_last", [&]{
        cpu_adaptive_avg_pool_channels_last<scalar_t, /* is_3d */ true>(output, input, output_size);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

void adapative_avg_pool3d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "adaptive_avg_pool3d_backward", [&] {
        cpu_adaptive_avg_pool_backward<scalar_t, /* is_3d */ true>(grad_input, grad_output);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "adaptive_avg_pool3d_backward_channels_last", [&]{
        cpu_adaptive_avg_pool_backward_channels_last<scalar_t, /* is_3d */ true>(grad_input, grad_output);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(adaptive_avg_pool2d_kernel, &adaptive_avg_pool2d_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(adaptive_avg_pool2d_backward_kernel, &adapative_avg_pool2d_backward_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(adaptive_avg_pool3d_kernel, &adaptive_avg_pool3d_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(adaptive_avg_pool3d_backward_kernel, &adapative_avg_pool3d_backward_kernel_impl);

}} // at::native

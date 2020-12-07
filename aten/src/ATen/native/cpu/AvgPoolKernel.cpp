#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/Pool.h>
#include <ATen/native/cpu/utils.h>

namespace at { namespace native {

namespace {

template <typename scalar_t>
void cpu_avg_pool(
    Tensor& output_,
    const Tensor& input_,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t numel = output.numel();
  int64_t ndim = input.ndimension();
  // treat batch size and channels as one dimension
  int64_t channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_height = output.size(-2);
  int64_t output_width = output.size(-1);

  // parallel on dim N, C, H, W
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t c = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, c, channels, oh, output_height, ow, output_width);

    for (int64_t i = begin; i < end; i++) {
      output_data[i] = static_cast<scalar_t>(0);

      // local pointers
      scalar_t* input_ptr = input_data + c * input_height * input_width;

      // compute the mean of the input image...
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      int64_t ih1 = std::min(ih0 + kH, input_height + padH);
      int64_t iw1 = std::min(iw0 + kW, input_width + padW);
      int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);
      ih0 = std::max(ih0, (int64_t) 0);
      iw0 = std::max(iw0, (int64_t) 0);
      ih1 = std::min(ih1, input_height);
      iw1 = std::min(iw1, input_width);

      if (ih0 >= ih1 || iw0 >= iw1) {
        // move on to next output index
        data_index_step(c, channels, oh, output_height, ow, output_width);
        continue;
      }

      scalar_t sum = 0;
    
      int64_t divide_factor;
      if (divisor_override.has_value()) {
        divide_factor = divisor_override.value();
      } else {
        if(count_include_pad) {
          divide_factor = pool_size;
        } else {
          divide_factor = (ih1 - ih0) * (iw1 - iw0);
        }
      }

      for (int64_t ih = ih0; ih < ih1; ih++) {
        for (int64_t iw = iw0; iw < iw1; iw++) {
          sum += input_ptr[ih * input_width + iw];
        }
      }
      output_data[i] += sum / divide_factor;

      // move on to next output index
      data_index_step(c, channels, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

template <typename scalar_t>
void cpu_avg_pool_channels_last(
    Tensor& output_,
    const Tensor& input_,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(input_.ndimension() == 4,
              "average pooling with channels last format supports tensors with 4 dims");
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output.size(2);
  int64_t output_width = output.size(3);

  using Vec = vec256::Vec256<scalar_t>;
  // parallel on dim N, H, W
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

    int64_t size = channels;
    int64_t len = size - (size % Vec::size());
    for (int64_t i = begin; i < end; i++) {
      // compute the mean of the input image...
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      int64_t ih1 = std::min(ih0 + kH, input_height + padH);
      int64_t iw1 = std::min(iw0 + kW, input_width + padW);
      int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);
      ih0 = std::max(ih0, (int64_t) 0);
      iw0 = std::max(iw0, (int64_t) 0);
      ih1 = std::min(ih1, input_height);
      iw1 = std::min(iw1, input_width);

      int64_t divide_factor;
      if (divisor_override.has_value()) {
        divide_factor = divisor_override.value();
      } else {
        if(count_include_pad) {
          divide_factor = pool_size;
        } else {
          divide_factor = (ih1 - ih0) * (iw1 - iw0);
        }
      }

      scalar_t* out = output_data + i * channels;

      // Pass I: zero the out lane
      int64_t d1 = 0;
      for (; d1 < len; d1 += Vec::size()) {
        Vec out_vec = Vec(scalar_t(0));
        out_vec.store(out + d1);
      }
      for (; d1 < size; d1++) {
        out[d1] = scalar_t(0);
      }

      if (ih0 >= ih1 || iw0 >= iw1) {
        // move on to next output index
        data_index_step(n, nbatch, oh, output_height, ow, output_width);
        continue;
      }

      // Pass II: compute local sum
      for (int64_t ih = ih0; ih < ih1; ih++) {
        for (int64_t iw = iw0; iw < iw1; iw++) {
          scalar_t* in = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;

          int64_t d2 = 0;
          for (; d2 < len; d2 += Vec::size()) {
            Vec out_vec = Vec::loadu(out + d2) + Vec::loadu(in + d2);
            out_vec.store(out + d2);
          }
          for (; d2 < size; d2++) {
            out[d2] += in[d2];
          }
        }
      }

      // Pass III: compute local average
      int64_t d3 = 0;
      for (; d3 < len; d3 += Vec::size()) {
        Vec out_vec = Vec::loadu(out + d3) / Vec(scalar_t(divide_factor));
        out_vec.store(out + d3);
      }
      for (; d3 < size; d3++) {
        out[d3] = out[d3] / divide_factor;
      }

      // move on to next output index
      data_index_step(n, nbatch, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t>
void cpu_avg_pool_backward(
    Tensor& grad_input_,
    const Tensor& grad_output_,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();

  int64_t ndim = grad_output.ndimension();
  // treat batch size and channels as one dimension
  int64_t channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // parallel on dim of N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      scalar_t* grad_input_ptr = grad_input_data + c * input_height * input_width;
      scalar_t* grad_output_ptr = grad_output_data + c * output_height * output_width;

      for (int64_t oh = 0; oh < output_height; oh++) {
        for (int64_t ow = 0; ow < output_width; ow++) {
          int64_t ih0 = oh * dH - padH;
          int64_t iw0 = ow * dW - padW;
          int64_t ih1 = std::min(ih0 + kH, input_height + padH);
          int64_t iw1 = std::min(iw0 + kW, input_width + padW);
          int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);
          ih0 = std::max(ih0, (int64_t) 0);
          iw0 = std::max(iw0, (int64_t) 0);
          ih1 = std::min(ih1, input_height);
          iw1 = std::min(iw1, input_width);

          int64_t divide_factor;
          if (divisor_override.has_value()) {
            divide_factor = divisor_override.value();
          } else {
            if(count_include_pad) {
              divide_factor = pool_size;
            } else {
              divide_factor = (ih1 - ih0) * (iw1 - iw0);
            }
          }

          scalar_t grad_delta = grad_output_ptr[oh * output_width + ow] / divide_factor;
          for (int64_t ih = ih0; ih < ih1; ih++) {
            for (int64_t iw = iw0; iw < iw1; iw++) {
              grad_input_ptr[ih * input_width + iw] += grad_delta;
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

template <typename scalar_t>
void cpu_avg_pool_backward_channels_last(
    Tensor& grad_input_,
    const Tensor& grad_output_,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);

  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto grad_output_data = grad_output.data_ptr<scalar_t>();

  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_height = grad_input.size(2);
  int64_t input_width = grad_input.size(3);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);

  using Vec = vec256::Vec256<scalar_t>;
  // parallel on dim N
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; n++) {
      scalar_t* grad_input_ptr = grad_input_data + n * input_height * input_width * channels;
      scalar_t* grad_output_ptr = grad_output_data + n * output_height * output_width * channels;

      for (int64_t oh = 0; oh < output_height; oh++) {
        for (int64_t ow = 0; ow < output_width; ow++) {
          int64_t ih0 = oh * dH - padH;
          int64_t iw0 = ow * dW - padW;
          int64_t ih1 = std::min(ih0 + kH, input_height + padH);
          int64_t iw1 = std::min(iw0 + kW, input_width + padW);
          int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);
          ih0 = std::max(ih0, (int64_t) 0);
          iw0 = std::max(iw0, (int64_t) 0);
          ih1 = std::min(ih1, input_height);
          iw1 = std::min(iw1, input_width);

          int64_t divide_factor;
          if (divisor_override.has_value()) {
            divide_factor = divisor_override.value();
          } else {
            if(count_include_pad) {
              divide_factor = pool_size;
            } else {
               divide_factor = (ih1 - ih0) * (iw1 - iw0);
            }
          }

          scalar_t* gout = grad_output_ptr + oh * output_width * channels + ow * channels;
          int64_t size = channels;
          int64_t len = size - (size % Vec::size());
          for (int64_t ih = ih0; ih < ih1; ih++) {
            for (int64_t iw = iw0; iw < iw1; iw++) {
              scalar_t* gin = grad_input_ptr + ih * input_width * channels + iw * channels;

              int64_t d = 0;
              for (; d < len; d += Vec::size()) {
                Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d) / Vec(scalar_t(divide_factor));
                gin_vec.store(gin + d);
              }
              for (; d < size; d++) {
                gin[d] += gout[d] / divide_factor;
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


void avg_pool2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(), "max_pool2d", [&] {
        cpu_avg_pool<scalar_t>(output, input, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(), "max_pool2d", [&] {
        cpu_avg_pool_channels_last<scalar_t>(output, input, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void avg_pool2d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, grad_output.scalar_type(), "avg_pool2d_backward", [&] {
        cpu_avg_pool_backward<scalar_t>(grad_input, grad_output, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, grad_output.scalar_type(), "avg_pool2d_backward", [&] {
        cpu_avg_pool_backward_channels_last<scalar_t>(grad_input, grad_output, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

} // anonymous namespace

REGISTER_DISPATCH(avg_pool2d_kernel, &avg_pool2d_kernel_impl);
REGISTER_DISPATCH(avg_pool2d_backward_kernel, &avg_pool2d_backward_kernel_impl);

}} // at::native

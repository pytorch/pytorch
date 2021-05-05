#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/Pool.h>
#include <ATen/native/cpu/utils.h>

namespace at { namespace native {

namespace {

template <typename scalar_t>
void cpu_max_pool(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();
  auto indices = indices_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

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
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
      int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
      while(ih0 < 0) { ih0 += dilationH; }
      while(iw0 < 0) { iw0 += dilationW; }

      // local pointers
      scalar_t* input_ptr = input_data + c * input_height * input_width;

      // compute local max
      int64_t maxindex = ih0 * input_width + iw0;
      scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
      for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
        for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
          int64_t index = ih * input_width + iw;
          scalar_t val = input_ptr[index];
          if ((val > maxval) || std::isnan(val)) {
            maxval = val;
            maxindex = index;
          }
        }
      }

      // set output to local max and store location of max
      output_data[i] = maxval;
      indices_data[i] = maxindex;

      // move on to next output index
      data_index_step(c, channels, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
  if (!indices_.is_contiguous()) {
    indices_.copy_(indices);
  }
}

template <typename scalar_t>
void cpu_max_pool_channels_last(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH) {
  TORCH_CHECK(input_.ndimension() == 4,
              "max pooling with channels last format supports tensors with 4 dims");
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output.size(2);
  int64_t output_width = output.size(3);

  using Vec = vec256::Vec256<scalar_t>;
  using integer_t = vec256::int_same_size_t<scalar_t>;
  using iVec = vec256::Vec256<integer_t>;
  // for the convience of vectorization, use integer of the same size of scalar_t,
  //   e.g. int32_t for float, int64_t for double
  // need to make sure doesn't overflow
  TORCH_CHECK(input_height <= std::ceil((double)std::numeric_limits<integer_t>::max() / (double)input_width));

  // parallel on dim N, H, W
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

    int64_t size = channels;
    int64_t len = size - (size % Vec::size());
    // temp buffer holding index with integer_t
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    std::unique_ptr<integer_t []> index_buffer(new integer_t[len]);

    for (int64_t i = begin; i < end; i++) {
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
      int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
      while(ih0 < 0) { ih0 += dilationH; }
      while(iw0 < 0) { iw0 += dilationW; }

      scalar_t* out = output_data + i * channels;
      int64_t* ind = indices_data + i * channels;

      // Pass I: init out lane
      iVec index0_vec = iVec(ih0 * input_width + iw0);
      Vec out_vec = Vec(-std::numeric_limits<scalar_t>::infinity());
      int64_t d1 = 0;
      for (; d1 < len; d1 += Vec::size()) {
        index0_vec.store(index_buffer.get() + d1);
        out_vec.store(out + d1);
      }
      for (; d1 < size; d1++) {
        ind[d1] = ih0 * input_width + iw0;
        out[d1] = -std::numeric_limits<scalar_t>::infinity();
      }
      // Pass II: compute local max
      for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
        for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
          scalar_t* in = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;

          int64_t d2 = 0;
          for (; d2 < len; d2 += Vec::size()) {
            iVec index_vec = iVec(ih * input_width + iw);
            Vec val_vec = Vec::loadu(in + d2);
            iVec maxindex_vec = iVec::loadu(index_buffer.get() + d2);
            Vec maxval_vec = Vec::loadu(out + d2);

            // true = all ones, false = all zeros
            Vec mask = (val_vec > maxval_vec) | val_vec.isnan();
            iVec imask = vec256::cast<integer_t>(mask);
            Vec out_vec = Vec::blendv(maxval_vec, val_vec, mask);
            iVec ind_vec = iVec::blendv(maxindex_vec, index_vec, imask);

            out_vec.store(out + d2);
            ind_vec.store(index_buffer.get() + d2);
          }
          for (; d2 < size; d2++) {
            int64_t index = ih * input_width + iw;
            scalar_t val = in[d2];
            int64_t maxindex = ind[d2];
            scalar_t maxval = out[d2];

            bool mask = (val > maxval) || std::isnan(val);
            out[d2] = mask ? val : maxval;
            ind[d2] = mask ? index : maxindex;
          }
        }
      }
      // convert indice data type
      vec256::convert<integer_t, int64_t>(index_buffer.get(), ind, len);

      // move on to next output index
      data_index_step(n, nbatch, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
  if (!indices_.is_contiguous(memory_format)) {
    indices_.copy_(indices);
  }
}

template <typename scalar_t>
void cpu_max_pool_backward(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& indices_) {
  auto grad_output = grad_output_.contiguous();
  auto indices = indices_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();
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
      int64_t * indices_ptr = indices_data + c * output_height * output_width;

      for (int64_t oh = 0; oh < output_height; oh++) {
        for (int64_t ow = 0; ow < output_width; ow++) {
          // retrieve position of max
          int64_t index = oh * output_width + ow;
          int64_t maxindex = indices_ptr[index];
          if (maxindex != -1) {
            // update gradient
            grad_input_ptr[maxindex] += grad_output_ptr[index];
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
void cpu_max_pool_backward_channels_last(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& indices_) {
  TORCH_CHECK(grad_output_.ndimension() == 4,
              "max pooling backward with channels last format supports tensors with 4 dims.");
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_height = grad_input.size(2);
  int64_t input_width = grad_input.size(3);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);

  // parallel on dim N
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; n++) {
      scalar_t* grad_input_ptr = grad_input_data + n * input_height * input_width * channels;
      scalar_t* grad_output_ptr = grad_output_data + n * output_height * output_width * channels;
      int64_t* indices_ptr = indices_data + n * output_height * output_width * channels;

      for (int64_t oh = 0; oh < output_height; oh++) {
        for (int64_t ow = 0; ow < output_width; ow++) {
          scalar_t* gout = grad_output_ptr + oh * output_width * channels + ow * channels;
          int64_t* ind = indices_ptr + oh * output_width * channels + ow * channels;
          // TODO: gcc vectorization
          for (int64_t c = 0; c < channels; c++) {
            int64_t maxindex = ind[c];
            if (maxindex != -1) {
              grad_input_ptr[maxindex * channels + c] += gout[c];
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

void max_pool2d_kernel_impl(
    const Tensor& output,
    const Tensor& indices,
    const Tensor& input,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d", [&] {
        cpu_max_pool<scalar_t>(output, indices, input, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_channels_last", [&] {
        cpu_max_pool_channels_last<scalar_t>(output, indices, input, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void max_pool2d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& indices) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "max_pool2d_backward", [&] {
        cpu_max_pool_backward<scalar_t>(grad_input, grad_output, indices);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "max_pool2d_backward_channels_last", [&] {
        cpu_max_pool_backward_channels_last<scalar_t>(grad_input, grad_output, indices);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(max_pool2d_kernel, &max_pool2d_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(max_pool2d_backward_kernel, &max_pool2d_backward_kernel_impl);

}} // at::native

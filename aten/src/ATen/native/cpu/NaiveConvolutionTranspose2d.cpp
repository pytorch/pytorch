#include <ATen/native/ConvUtils.h>
#include <ATen/native/NaiveConvolutionTranspose2d.h>

namespace at {
namespace native {

namespace {
static void slow_conv_transpose2d_backward_out_cpu_template(
    const Tensor& input_,
    const Tensor& grad_output_,
    Tensor& grad_input,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  TORCH_CHECK(
      output_padding.size() == 2,
      "It is expected stride equals to 2, but got size ",
      output_padding.size());

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];
  int64_t output_padding_height = output_padding[0];
  int64_t output_padding_width = output_padding[1];

  int64_t n_input_plane = weight_.size(0);
  int64_t n_output_plane = weight_.size(1);

  slow_conv_transpose2d_shape_check(
      input_,
      grad_output_,
      weight_,
      Tensor(),
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      output_padding_height,
      output_padding_width,
      dilation_height,
      dilation_width,
      false);

  Tensor input = input_.contiguous();
  Tensor grad_output = grad_output_.contiguous();
  Tensor weight = weight_.contiguous();

  bool is_batch = false;
  if (input.dim() == 3) {
    // Force batch
    is_batch = true;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
    grad_output.resize_(
        {1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});
  }

  int64_t input_width = input.size(3);
  int64_t input_height = input.size(2);
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // Batch size + input planes
  int64_t batch_size = input.size(0);

  // Resize output
  grad_input.resize_({batch_size, n_input_plane, input_height, input_width});
  grad_input.zero_();

  // Create temporary columns
  bool need_columns = (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
      stride_width != 1 || pad_height != 0 || pad_width != 0 ||
      dilation_height != 1 || dilation_width != 1);
  Tensor grad_columns = need_columns ? at::empty({n_output_plane * kernel_width * kernel_height,
      input_height * input_width}, input.options()) : Tensor();

  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "slow_conv_transpose2d_backward_out_cpu", [&] {
        // Helpers
        Tensor grad_input_n = Tensor();
        Tensor grad_output_n = Tensor();

        // For each elt in batch, do:
        for (const auto elt : c10::irange(batch_size)) {
          // Matrix mulitply per sample:
          grad_input_n = grad_input.select(0, elt);
          grad_output_n = grad_output.select(0, elt);

          if (need_columns) {
            // Extract columns:
            im2col<scalar_t>(
                  grad_output_n.data_ptr<scalar_t>(),
                  n_output_plane,
                  output_height,
                  output_width,
                  input_height,
                  input_width,
                  kernel_height,
                  kernel_width,
                  pad_height,
                  pad_width,
                  stride_height,
                  stride_width,
                  dilation_height,
                  dilation_width,
                  grad_columns.data_ptr<scalar_t>());
          }

          // M,N,K are dims of matrix A and B
          // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
          int64_t m = weight.size(0);
          int64_t n = input_height * input_width;
          int64_t k = weight.size(1) * weight.size(2) * weight.size(3);

          // Do GEMM (note: this is a bit confusing because gemm assumes
          // column-major matrices)
          auto gemm_in_ptr = need_columns ? grad_columns.data_ptr<scalar_t>()
              : grad_output_n.data_ptr<scalar_t>();
          cpublas::gemm(
              TransposeType::NoTranspose,
              TransposeType::NoTranspose,
              n,
              m,
              k,
              1,
              gemm_in_ptr,
              n,
              weight.data_ptr<scalar_t>(),
              k,
              0,
              grad_input_n.data_ptr<scalar_t>(),
              n);
        }

        // Resize output
        if (is_batch) {
          grad_output.resize_({n_output_plane, output_height, output_width});
          input.resize_({n_input_plane, input_height, input_width});
          grad_input.resize_({n_input_plane, input_height, input_width});
        }
      });
}

void slow_conv_transpose2d_acc_grad_parameters_cpu(
    const Tensor& input_,
    const Tensor& grad_output_,
    Tensor& grad_weight,
    Tensor& grad_bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    int scale_) {
  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  TORCH_CHECK(
      output_padding.size() == 2,
      "It is expected stride equals to 2, but got size ",
      output_padding.size());

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];
  int64_t output_padding_height = output_padding[0];
  int64_t output_padding_width = output_padding[1];

  slow_conv_transpose2d_shape_check(
      input_,
      grad_output_,
      grad_weight,
      grad_bias,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      output_padding_height,
      output_padding_width,
      dilation_height,
      dilation_width,
      true);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t n_output_plane;
  if (grad_weight.defined()) {
    n_output_plane = grad_weight.size(1);
  } else if (grad_bias.defined()) {
    n_output_plane = grad_bias.size(0);
  } else {
    return;
  }

  Tensor input = input_.contiguous();
  Tensor grad_output = grad_output_.contiguous();

  if (grad_weight.defined()) {
    TORCH_CHECK(
        grad_weight.is_contiguous(), "grad_weight needs to be contiguous");
  }
  if (grad_bias.defined()) {
    TORCH_CHECK(grad_bias.is_contiguous(), "grad_bias needs to be contiguous");
  }

  bool is_batch = false;
  if (input.dim() == 3) {
    // Force batch
    is_batch = true;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
    grad_output.resize_(
        {1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});
  }

  int64_t input_width = input.size(3);
  int64_t input_height = input.size(2);
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // Batch size + input planes
  int64_t batch_size = input.size(0);

  // Resize temporary columns
  bool need_columns = (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
      stride_width != 1 || pad_height != 0 || pad_width != 0 ||
      dilation_height != 1 || dilation_width != 1);
  Tensor columns = need_columns ? at::empty({n_output_plane * kernel_width * kernel_height,
      input_height * input_width}, input.options()) : Tensor();

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "slow_conv_transpose2d_acc_grad_parameters_cpu", [&] {
        // Helpers
        Tensor input_n = Tensor();
        Tensor grad_output_n = Tensor();

        scalar_t scale = static_cast<scalar_t>(scale_);

        // For each elt in batch, do:
        for (const auto elt : c10::irange(batch_size)) {
          // Matrix mulitply per output:
          grad_output_n = grad_output.select(0, elt);

          // Do Weight:
          if (grad_weight.defined()) {
            // Matrix mulitply per output:
            input_n = input.select(0, elt);

            if (need_columns) {
              // Extract columns:
              im2col<scalar_t>(
                  grad_output_n.data_ptr<scalar_t>(),
                  n_output_plane,
                  output_height,
                  output_width,
                  input_height,
                  input_width,
                  kernel_height,
                  kernel_width,
                  pad_height,
                  pad_width,
                  stride_height,
                  stride_width,
                  dilation_height,
                  dilation_width,
                  columns.data_ptr<scalar_t>());
            }

            // M,N,K are dims of matrix A and B
            // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
            int64_t n = n_output_plane * kernel_height * kernel_width;
            int64_t m = input_n.size(0); // n_input_plane
            int64_t k = input_height * input_width;

            // Do GEMM (note: this is a bit confusing because gemm assumes
            // column-major matrices)
            auto gemm_in_ptr = need_columns ? columns.data_ptr<scalar_t>()
                : grad_output_n.data_ptr<scalar_t>();
            cpublas::gemm(
                TransposeType::Transpose,
                TransposeType::NoTranspose,
                n,
                m,
                k,
                scale,
                gemm_in_ptr,
                k,
                input_n.data_ptr<scalar_t>(),
                k,
                1,
                grad_weight.data_ptr<scalar_t>(),
                n);
          }
        }

        if (grad_bias.defined()) {
          at::sum_out(grad_bias, grad_output, IntArrayRef{0, 2, 3});
        }

        // Resize
        if (is_batch) {
          grad_output.resize_({n_output_plane, output_height, output_width});
          input.resize_({input.size(1), input_height, input_width});
        }
      });
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv_transpose2d_backward_out_cpu(const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  if (grad_input.defined()) {
    slow_conv_transpose2d_backward_out_cpu_template(
        input,
        grad_output,
        grad_input,
        weight,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation);
  }

  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes());
    grad_weight.zero_();
  }

  if (grad_bias.defined()) {
    grad_bias.resize_({weight.size(1)});
    grad_bias.zero_();
  }

  if (grad_weight.defined() || grad_bias.defined()) {
    slow_conv_transpose2d_acc_grad_parameters_cpu(
        input,
        grad_output,
        grad_weight,
        grad_bias,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        1);
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>(
      grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> slow_conv_transpose2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    std::array<bool, 3> output_mask) {
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  } else {
    grad_input = Tensor();
  }

  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  } else {
    grad_weight = Tensor();
  }

  if (output_mask[2]) {
    grad_bias = at::empty({0}, grad_output.options());
  } else {
    grad_bias = Tensor();
  }

  if (grad_input.defined()) {
    slow_conv_transpose2d_backward_out_cpu_template(
        input,
        grad_output,
        grad_input,
        weight,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation);
  }

  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes());
    grad_weight.zero_();
  }

  if (grad_bias.defined()) {
    grad_bias.resize_({weight.size(1)});
    grad_bias.zero_();
  }

  if (grad_weight.defined() || grad_bias.defined()) {
    slow_conv_transpose2d_acc_grad_parameters_cpu(
        input,
        grad_output,
        grad_weight,
        grad_bias,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        1);
  }

  return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}

} // namespace

REGISTER_DISPATCH(slow_conv_transpose2d_backward_stub, &slow_conv_transpose2d_backward_cpu);

} // namespace native
} // namespace at

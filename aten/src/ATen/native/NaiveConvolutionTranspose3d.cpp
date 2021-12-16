#include <ATen/native/NaiveConvolutionTranspose3d.h>

namespace at {
namespace native {
namespace {

void slow_conv_transpose3d_out_cpu_template(
    Tensor& output,
    const Tensor& input_, // 4D or 5D (batch) tensor
    const Tensor& weight_, // weight tensor (n_input_plane x n_output_plane x
                           // kernel_depth x kernel_height x kernel_width)
    IntArrayRef kernel_size,
    const Tensor& bias_,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  TORCH_CHECK(
      kernel_size.size() == 3,
      "It is expected kernel_size equals to 3, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 3,
      "It is expected dilation equals to 3, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 3,
      "It is expected padding equals to 3, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 3,
      "It is expected stride equals to 3, but got size ",
      stride.size());

  TORCH_CHECK(
      output_padding.size() == 3,
      "It is expected stride equals to 3, but got size ",
      output_padding.size());

  int64_t kernel_depth = kernel_size[0];
  int64_t kernel_height = kernel_size[1];
  int64_t kernel_width = kernel_size[2];
  int64_t dilation_depth = dilation[0];
  int64_t dilation_height = dilation[1];
  int64_t dilation_width = dilation[2];
  int64_t padding_depth = padding[0];
  int64_t padding_height = padding[1];
  int64_t padding_width = padding[2];
  int64_t stride_depth = stride[0];
  int64_t stride_height = stride[1];
  int64_t stride_width = stride[2];
  int64_t output_padding_depth = output_padding[0];
  int64_t output_padding_height = output_padding[1];
  int64_t output_padding_width = output_padding[2];

  slow_conv_transpose3d_shape_check(
      input_,
      Tensor(),
      weight_,
      bias_,
      kernel_depth,
      kernel_width,
      kernel_height,
      stride_depth,
      stride_width,
      stride_height,
      padding_depth,
      padding_width,
      padding_height,
      dilation_depth,
      dilation_width,
      dilation_height,
      output_padding_depth,
      output_padding_width,
      output_padding_height,
      0);

  Tensor input = input_.contiguous();
  Tensor weight = weight_.contiguous();
  Tensor bias = bias_.defined() ? bias_.contiguous() : bias_;

  const int n_input_plane = (int)weight.size(0);
  const int n_output_plane = (int)weight.size(1);

  bool is_batch = false;
  if (input.dim() == 4) {
    // Force batch
    is_batch = true;
    input.resize_(
        {1, input.size(0), input.size(1), input.size(2), input.size(3)});
  }

  const int64_t input_width = input.size(4);
  const int64_t input_height = input.size(3);
  const int64_t input_depth = input.size(2);

  const int64_t output_depth = (input_depth - 1) * stride_depth -
      2 * padding_depth + (dilation_depth * (kernel_depth - 1) + 1) +
      output_padding_depth;
  const int64_t output_height = (input_height - 1) * stride_height -
      2 * padding_height + (dilation_height * (kernel_height - 1) + 1) +
      output_padding_height;
  const int64_t output_width = (input_width - 1) * stride_width -
      2 * padding_width + (dilation_width * (kernel_width - 1) + 1) +
      output_padding_width;

  // Batch size + input planes
  const int64_t batch_size = input.size(0);

  // Resize output
  output.resize_(
      {batch_size, n_output_plane, output_depth, output_height, output_width});

  // Create temporary columns
  Tensor columns = at::empty({n_output_plane * kernel_width * kernel_height * kernel_depth,
      input_depth * input_height * input_width}, input.options());

  // Define a buffer of ones, for bias accumulation
  Tensor ones = bias.defined() ? at::ones({output_depth, output_height, output_width}, input_.options()) : Tensor();

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long,
      input.scalar_type(), "slow_conv_transpose3d_out_cpu", [&] {
        // Helpers
        Tensor input_n;
        Tensor output_n;

        int64_t elt;
        // For each elt in batch, do:
        for (elt = 0; elt < batch_size; ++elt) {
          // Matrix mulitply per output:
          input_n = input.select(0, elt);
          output_n = output.select(0, elt);

          // M,N,K are dims of matrix A and B
          // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
          const int64_t m =
              weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4);
          const int64_t n = columns.size(1);
          const int64_t k = weight.size(0);

          // Do GEMM (note: this is a bit confusing because gemm assumes
          // column-major matrices)
          cpublas::gemm(
              TransposeType::NoTranspose,
              TransposeType::Transpose,
              n,
              m,
              k,
              1,
              input_n.data_ptr<scalar_t>(),
              n,
              weight.data_ptr<scalar_t>(),
              m,
              0,
              columns.data_ptr<scalar_t>(),
              n);

          // Unpack columns back into input:
          at::native::col2vol<scalar_t>(
              columns.data_ptr<scalar_t>(),
              n_output_plane,
              output_depth,
              output_height,
              output_width,
              input_depth,
              input_height,
              input_width,
              kernel_depth,
              kernel_height,
              kernel_width,
              padding_depth,
              padding_height,
              padding_width,
              stride_depth,
              stride_height,
              stride_width,
              dilation_depth,
              dilation_height,
              dilation_width,
              output_n.data_ptr<scalar_t>());

          // Do Bias after:
          // M,N,K are dims of matrix A and B
          // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
          const int64_t m_ = n_output_plane;
          const int64_t n_ = output_depth * output_height * output_width;
          const int64_t k_ = 1;

          // Do GEMM (note: this is a bit confusing because gemm assumes
          // column-major matrices)
          if (bias.defined()) {
            cpublas::gemm(
                TransposeType::Transpose,
                TransposeType::NoTranspose,
                n_,
                m_,
                k_,
                1,
                ones.data_ptr<scalar_t>(),
                k_,
                bias.data_ptr<scalar_t>(),
                k_,
                1,
                output_n.data_ptr<scalar_t>(),
                n_);
          }
        }

        // Resize output
        if (is_batch) {
          output.resize_(
              {n_output_plane, output_depth, output_height, output_width});
          input.resize_(
              {n_input_plane, input_depth, input_height, input_width});
        }
      });
}

} // namespace

Tensor& slow_conv_transpose3d_out_cpu(const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  slow_conv_transpose3d_out_cpu_template(
      output,
      input,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output_padding,
      dilation);

  return output;
}

Tensor slow_conv_transpose3d_cpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  slow_conv_transpose3d_out_cpu_template(
      output,
      input,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output_padding,
      dilation);

  return output;
}

} // namespace native
} // namespace at

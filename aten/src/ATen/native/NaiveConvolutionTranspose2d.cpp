#include <ATen/native/NaiveConvolutionTranspose2d.h>

namespace at {
namespace meta {
TORCH_META_FUNC(slow_conv_transpose2d)
(const Tensor& input,
 const Tensor& weight,
 IntArrayRef kernel_size,
 OptionalTensorRef bias_opt,
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

  slow_conv_transpose2d_shape_check(
      input,
      Tensor(),
      weight,
      bias_opt.getTensorRef(),
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

  int n_output_plane = weight.size(1);

  Tensor input_ = input.contiguous();

  if (input_.dim() == 3) {
    input_.resize_({1, input_.size(0), input_.size(1), input_.size(2)});
  }

  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // Batch size + input planes
  int64_t batch_size = input_.size(0);

  // Resize output
  TensorOptions options(input.options());
  set_output(
      {batch_size, n_output_plane, output_height, output_width},
      options.memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
}
} // namespace meta

namespace native {

template<typename scalar_t>
void gemv(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);

namespace {
void slow_conv_transpose2d_out_cpu_template(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
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

  int n_input_plane = weight.size(0);
  int n_output_plane = weight.size(1);

  Tensor input_ = input.contiguous();
  Tensor weight_ = weight.contiguous();

  Tensor bias_ = Tensor();

  if (bias.defined()) {
    bias_ = bias.contiguous();
  }

  bool is_batch = false;
  if (input_.dim() == 3) {
    // Force batch
    is_batch = true;
  }

  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // Batch size + input planes
  int64_t batch_size = input_.size(0);

  // Create temporary columns
  Tensor columns = at::zeros({n_output_plane * kernel_width * kernel_height,
      input_height * input_width}, input_.options());

  // Define a buffer of ones, for bias accumulation
  Tensor ones = bias.defined() ? at::ones({output_height, output_width}, input_.options()) : Tensor();

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long,
      input.scalar_type(), "slow_conv_transpose2d_out_cpu", [&] {
        // For each elt in batch, do:
        for (const auto elt : c10::irange(batch_size)) {
          // Helpers
          Tensor input_n;
          Tensor output_n;

          // Matrix mulitply per output:
          input_n = input_.select(0, elt);
          output_n = output.select(0, elt);

          // M,N,K are dims of matrix A and B
          // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
          int64_t m = weight_.size(1) * weight_.size(2) * weight_.size(3);
          int64_t n = input_height * input_width;
          int64_t k = weight_.size(0);

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
              weight_.data_ptr<scalar_t>(),
              m,
              0,
              columns.data_ptr<scalar_t>(),
              n);

          // Unpack columns back into input:
          col2im<scalar_t>(
              columns.data_ptr<scalar_t>(),
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
              output_n.data_ptr<scalar_t>());

          // Do Bias after:
          // M,N,K are dims of matrix A and B
          // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
          int64_t m_ = n_output_plane;
          int64_t n_ = output_height * output_width;
          int64_t k_ = 1;

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
                bias_.data_ptr<scalar_t>(),
                k_,
                1,
                output_n.data_ptr<scalar_t>(),
                n_);
          }
        }

        // Resize output
        if (is_batch) {
          output.resize_({n_output_plane, output_height, output_width});
          input_.resize_({n_input_plane, input_height, input_width});
        }
      });
}
} // namespace

TORCH_IMPL_FUNC(slow_conv_transpose2d_structured_cpu)
(const Tensor& input,
 const Tensor& weight,
 IntArrayRef kernel_size,
 OptionalTensorRef bias_opt,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef output_padding,
 IntArrayRef dilation,
 const Tensor& output){
  const Tensor& bias = bias_opt.getTensorRef();

  slow_conv_transpose2d_out_cpu_template(
      output,
      input,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output_padding,
      dilation);
 }

} // namespace native
} // namespace at

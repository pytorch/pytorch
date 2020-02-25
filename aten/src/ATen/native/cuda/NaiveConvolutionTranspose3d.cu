#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <ATen/native/cuda/vol2col.cuh>

namespace at {
namespace native {
namespace {

static inline void slow_conv_transpose3d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& bias,
    int kernel_depth,
    int kernel_width,
    int kernel_height,
    int stride_depth,
    int stride_width,
    int stride_height,
    int padding_depth,
    int padding_width,
    int padding_height,
    int dilation_depth,
    int dilation_width,
    int dilation_height,
    int output_padding_depth,
    int output_padding_width,
    int output_padding_height,
    int weight_nullable) {
  TORCH_CHECK(
      input.numel() != 0 && (input.dim() == 4 || input.dim() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input, but got: ",
      input.sizes());
  TORCH_CHECK(
      stride_depth > 0 && stride_width > 0 && stride_height > 0,
      "stride should be greater than zero, but got stride_depth: ",
      stride_depth,
      " stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width);
  TORCH_CHECK(
      dilation_depth > 0 && dilation_width > 0 && dilation_height > 0,
      "dilation should be greater than zero, but got dilation_depth: ",
      dilation_depth,
      ", dilation_height: ",
      dilation_height,
      ", dilation_width: ",
      dilation_width);
  TORCH_CHECK(
      (output_padding_depth < stride_depth ||
       output_padding_depth < dilation_depth) &&
          (output_padding_width < stride_width ||
           output_padding_width < dilation_width) &&
          (output_padding_height < stride_height ||
           output_padding_height < dilation_height),
      "output padding must be smaller than either stride or dilation,",
      " but got output_padding_depth: ",
      output_padding_depth,
      " output_padding_height: ",
      output_padding_height,
      " output_padding_width: ",
      output_padding_width,
      " stride_depth: ",
      stride_depth,
      " stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width,
      " dilation_depth: ",
      dilation_depth,
      " dilation_height: ",
      dilation_height,
      " dilation_width: ",
      dilation_width);

  // number of input & output planes and kernel size is indirectly defined by
  // the weight tensor
  if (weight.defined()) {
    TORCH_CHECK(
        weight.numel() != 0 && weight.dim() == 5,
        "non-empty 5D (n_output_plane x n_input_plane ",
        "x kernel_depth x kernel_height x kernel_width) tensor ",
        "expected for weight, but got: ",
        weight.sizes());
    if (bias.defined()) {
      check_dim_size(bias, 1, 0, weight.size(1));
    }
  } else if (!weight_nullable) {
    AT_ERROR("weight tensor is expected to be non-nullable");
  }

  int ndim = input.dim();
  int dimf = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;

  if (ndim == 5) {
    dimf++;
    dimd++;
    dimh++;
    dimw++;
  }

  if (weight.defined()) {
    const int64_t n_input_plane = weight.size(0);
    check_dim_size(input, ndim, dimf, n_input_plane);
  }

  int64_t input_width = input.size(dimw);
  int64_t input_height = input.size(dimh);
  int64_t input_depth = input.size(dimd);

  int64_t output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth +
      (dilation_depth * (kernel_depth - 1) + 1) + output_padding_depth;
  int64_t output_height = (input_height - 1) * stride_height -
      2 * padding_height + (dilation_height * (kernel_height - 1) + 1) +
      output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * padding_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  if (output_depth < 1 || output_width < 1 || output_height < 1) {
    AT_ERROR(
        "Given input size per channel: (",
        input_depth,
        " x ",
        input_height,
        " x ",
        input_width,
        "). Calculated output size per channel: (",
        output_depth,
        " x ",
        output_height,
        " x ",
        output_width,
        "). Output size is too small");
  }

  if (grad_output.defined()) {
    if (weight.defined()) {
      const int64_t n_output_plane = weight.size(1);
      check_dim_size(grad_output, ndim, dimf, n_output_plane);
    } else if (bias.defined()) {
      const int64_t n_output_plane = bias.size(0);
      check_dim_size(grad_output, ndim, dimf, n_output_plane);
    }
    check_dim_size(grad_output, ndim, dimd, output_depth);
    check_dim_size(grad_output, ndim, dimh, output_height);
    check_dim_size(grad_output, ndim, dimw, output_width);
  }
}

void slow_conv_transpose3d_out_cuda_template(
    Tensor& output,
    const Tensor& input_,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& finput,
    Tensor& fgrad_input) {
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

  Tensor columns = finput;
  Tensor ones = fgrad_input;

  int n_input_plane = weight_.size(0);
  int n_output_plane = weight_.size(1);

  TensorArg input_arg{input_, "input", 1}, output_arg{output, "output", 2},
      weight_arg{weight_, "weight", 3}, bias_arg{bias, "bias", 4},
      columns_arg{columns, "columns", 5}, ones_arg{ones, "ones", 6};

  checkAllSameGPU(
      "slow_conv_transpose3d_out_cuda",
      {input_arg, output_arg, weight_arg, bias_arg, columns_arg, ones_arg});

  slow_conv_transpose3d_shape_check(
      input_,
      Tensor(),
      weight_,
      bias,
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

  TORCH_CHECK(
      !bias.defined() || bias.is_contiguous(),
      "bias tensor has to be contiguous");

  Tensor input = input_.contiguous();
  Tensor weight = weight_.contiguous();

  int is_batch = false;
  if (input.dim() == 4) {
    // Force batch
    is_batch = true;
    input.resize_(
        {1, input.size(0), input.size(1), input.size(2), input.size(3)});
  }

  int64_t input_width = input.size(4);
  int64_t input_height = input.size(3);
  int64_t input_depth = input.size(2);

  int64_t output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth +
      (dilation_depth * (kernel_depth - 1) + 1) + output_padding_depth;
  int64_t output_height = (input_height - 1) * stride_height -
      2 * padding_height + (dilation_height * (kernel_height - 1) + 1) +
      output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * padding_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // Batch size + input planes
  int64_t batch_size = input.size(0);

  // Resize output
  output.resize_(
      {batch_size, n_output_plane, output_depth, output_height, output_width});

  // Resize temporary columns
  columns.resize_({n_output_plane * kernel_width * kernel_height * kernel_depth,
                   input_depth * input_height * input_width});

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets
  // increased, and always contains ones.
  if (ones.dim() != 3 ||
      ones.size(0) * ones.size(1) * ones.size(2) <
          output_depth * output_height * output_width) {
    // Resize plane and fill with ones...
    ones.resize_({output_depth, output_height, output_width});
    ones.fill_(1);
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
      input.scalar_type(), "slow_conv_transpose3d_out_cuda", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        // Helpers
        Tensor input_n;
        Tensor output_n;

        // For each elt in batch, do:
        for (int elt = 0; elt < batch_size; elt++) {
          // Matrix mulitply per output:
          input_n = input.select(0, elt);
          output_n = output.select(0, elt);

          // M,N,K are dims of matrix A and B
          // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
          int64_t m =
              weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4);
          int64_t n = columns.size(1);
          int64_t k = weight.size(0);

          // Do GEMM (note: this is a bit confusing because gemm assumes
          // column-major matrices)
          at::cuda::blas::gemm<scalar_t>(
              'n',
              't',
              n,
              m,
              k,
              static_cast<scalar_t>(1),
              input_n.data_ptr<scalar_t>(),
              n,
              weight.data_ptr<scalar_t>(),
              m,
              static_cast<scalar_t>(0),
              columns.data_ptr<scalar_t>(),
              n);

          // Unpack columns back into input:
          at::native::col2vol<scalar_t, accscalar_t>(
              at::cuda::getCurrentCUDAStream(),
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
          int64_t m_ = n_output_plane;
          int64_t n_ = output_depth * output_height * output_width;
          int64_t k_ = 1;

          // Do GEMM (note: this is a bit confusing because gemm assumes
          // column-major matrices)
          if (bias.defined()) {
            at::cuda::blas::gemm<scalar_t>(
                't',
                'n',
                n_,
                m_,
                k_,
                static_cast<scalar_t>(1),
                ones.data_ptr<scalar_t>(),
                k_,
                bias.data_ptr<scalar_t>(),
                k_,
                static_cast<scalar_t>(1),
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

void slow_conv_transpose3d_backward_out_cuda_template(
    const Tensor& input_,
    const Tensor& grad_output_,
    Tensor& grad_input,
    const Tensor& weight_,
    const Tensor& finput,
    const Tensor& fgrad_input,
    IntArrayRef kernel_size,
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

  Tensor grad_columns = finput;

  int n_input_plane = weight_.size(0);
  int n_output_plane = weight_.size(1);

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

  TensorArg input_arg{input_, "input", 1},
      grad_output_arg{grad_output_, "grad_output", 2},
      weight_arg{weight_, "weight", 3},
      grad_columns_arg{grad_columns, "grad_columns", 4},
      grad_input_arg{grad_input, "grad_input", 5};

  checkAllSameGPU(
      "slow_conv_transpose3d_backward_out_cuda",
      {input_arg,
       grad_output_arg,
       weight_arg,
       grad_columns_arg,
       grad_input_arg});

  slow_conv_transpose3d_shape_check(
      input_,
      grad_output_,
      weight_,
      Tensor(),
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
  Tensor grad_output = grad_output_.contiguous();
  Tensor weight = weight_.contiguous();

  bool is_batch = false;
  if (input.dim() == 4) {
    // Force batch
    is_batch = true;
    input.resize_(
        {1, input.size(0), input.size(1), input.size(2), input.size(3)});
    grad_output.resize_({1,
                         grad_output.size(0),
                         grad_output.size(1),
                         grad_output.size(2),
                         grad_output.size(3)});
  }

  int64_t input_width = input.size(4);
  int64_t input_height = input.size(3);
  int64_t input_depth = input.size(2);
  int64_t output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth +
      (dilation_depth * (kernel_depth - 1) + 1) + output_padding_depth;
  int64_t output_height = (input_height - 1) * stride_height -
      2 * padding_height + (dilation_height * (kernel_height - 1) + 1) +
      output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * padding_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // Batch size + input planes
  int64_t batch_size = input.size(0);

  // Resize output
  grad_input.resize_(
      {batch_size, n_input_plane, input_depth, input_height, input_width});

  // Resize temporary columns
  grad_columns.resize_(
      {n_output_plane * kernel_width * kernel_height * kernel_depth,
       input_depth * input_height * input_width});

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
      input.scalar_type(), "slow_conv_transpose3d_backward_out_cuda", [&] {
        // Helpers
        Tensor grad_input_n;
        Tensor grad_output_n;

        // For each elt in batch, do:
        for (int elt = 0; elt < batch_size; elt++) {
          // Matrix mulitply per sample:
          grad_input_n = grad_input.select(0, elt);
          grad_output_n = grad_output.select(0, elt);

          // Extract columns:
          at::native::vol2col<scalar_t>(
              at::cuda::getCurrentCUDAStream(),
              grad_output_n.data_ptr<scalar_t>(),
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
              grad_columns.data_ptr<scalar_t>());

          // M,N,K are dims of matrix A and B
          // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
          int64_t m = weight.size(0);
          int64_t n = grad_columns.size(1);
          int64_t k =
              weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4);

          // Do GEMM (note: this is a bit confusing because gemm assumes
          // column-major matrices)
          at::cuda::blas::gemm<scalar_t>(
              'n',
              'n',
              n,
              m,
              k,
              static_cast<scalar_t>(1),
              grad_columns.data_ptr<scalar_t>(),
              n,
              weight.data_ptr<scalar_t>(),
              k,
              static_cast<scalar_t>(0),
              grad_input_n.data_ptr<scalar_t>(),
              n);
        }

        // Resize output
        if (is_batch) {
          grad_output.resize_(
              {n_output_plane, output_depth, output_height, output_width});
          input.resize_(
              {n_input_plane, input_depth, input_height, input_width});
          grad_input.resize_(
              {n_input_plane, input_depth, input_height, input_width});
        }
      });
}

void slow_conv_transpose3d_acc_grad_parameters_cuda(
    const Tensor& input_,
    const Tensor& grad_output_,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& finput,
    const Tensor& fgrad_input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    int scale_) {
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

  Tensor columns = finput;
  Tensor ones = fgrad_input;

  TensorArg input_arg{input_, "input", 1},
      grad_output_arg{grad_output_, "grad_output", 2},
      grad_weight_arg{grad_weight, "grad_weight", 3},
      grad_bias_arg{grad_bias, "grad_bias", 4},
      columns_arg{columns, "columns", 5}, ones_arg{ones, "ones", 6};

  checkAllSameGPU(
      "slow_conv_transpose3d_acc_grad_parameters_cuda",
      {input_arg,
       grad_output_arg,
       grad_weight_arg,
       grad_bias_arg,
       columns_arg,
       ones_arg});

  slow_conv_transpose3d_shape_check(
      input_,
      grad_output_,
      grad_weight,
      grad_bias,
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
      1);

  int n_output_plane;
  if (grad_weight.defined()) {
    n_output_plane = grad_weight.size(1);
  } else if (grad_bias.defined()) {
    n_output_plane = grad_bias.size(0);
  } else {
    return;
  }

  if (grad_weight.defined()) {
    TORCH_CHECK(
        grad_weight.is_contiguous(), "grad_weight needs to be contiguous");
  }
  if (grad_bias.defined()) {
    TORCH_CHECK(grad_bias.is_contiguous(), "grad_bias needs to be contiguous");
    TORCH_CHECK(ones.is_contiguous(), "ones needs to be contiguous");
  }

  Tensor input = input_.contiguous();
  Tensor grad_output = grad_output_.contiguous();

  bool is_batch = false;
  if (input.dim() == 4) {
    // Force batch
    is_batch = true;
    input.resize_(
        {1, input.size(0), input.size(1), input.size(2), input.size(3)});
    grad_output.resize_({1,
                         grad_output.size(0),
                         grad_output.size(1),
                         grad_output.size(2),
                         grad_output.size(3)});
  }

  int64_t input_width = input.size(4);
  int64_t input_height = input.size(3);
  int64_t input_depth = input.size(2);

  int64_t output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth +
      (dilation_depth * (kernel_depth - 1) + 1) + output_padding_depth;
  int64_t output_height = (input_height - 1) * stride_height -
      2 * padding_height + (dilation_height * (kernel_height - 1) + 1) +
      output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * padding_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // Batch size + input planes
  int64_t batch_size = input.size(0);

  // Define a buffer of ones, for bias accumulation
  if (ones.dim() != 3 ||
      ones.size(0) * ones.size(1) * ones.size(2) <
          output_depth * output_height * output_width) {
    // Resize plane and fill with ones...
    ones.resize_({output_depth, output_height, output_width});
    ones.fill_(1);
  }

  // Resize temporary columns
  columns.resize_({n_output_plane * kernel_width * kernel_height * kernel_depth,
                   input_depth * input_height * input_width});

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
      input.scalar_type(),
      "slow_conv_transpose3d_acc_grad_parameters_cuda",
      [&] {
        // Helpers
        Tensor input_n;
        Tensor grad_output_n;

        scalar_t scale = static_cast<scalar_t>(scale_);

        // For each elt in batch, do:
        for (int elt = 0; elt < batch_size; elt++) {
          // Matrix mulitply per output:
          grad_output_n = grad_output.select(0, elt);

          // Do Weight:
          if (grad_weight.defined()) {
            // Matrix mulitply per output:
            input_n = input.select(0, elt);

            // Extract columns:
            at::native::vol2col<scalar_t>(
                at::cuda::getCurrentCUDAStream(),
                grad_output_n.data_ptr<scalar_t>(),
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
                columns.data_ptr<scalar_t>());

            // M,N,K are dims of matrix A and B
            // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
            int64_t n = columns.size(0); // n_output_plane * kt * kh * kw
            int64_t m = input_n.size(0); // n_input_plane
            int64_t k = columns.size(1); // input_height * input_width

            // Do GEMM (note: this is a bit confusing because gemm assumes
            // column-major matrices)
            at::cuda::blas::gemm<scalar_t>(
                't',
                'n',
                n,
                m,
                k,
                scale,
                columns.data_ptr<scalar_t>(),
                k,
                input_n.data_ptr<scalar_t>(),
                k,
                static_cast<scalar_t>(1),
                grad_weight.data_ptr<scalar_t>(),
                n);
          }

          // Do Bias:
          if (grad_bias.defined()) {
            // M,N,K are dims of matrix A and B
            // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
            int64_t m_ = n_output_plane;
            int64_t k_ = output_depth * output_height * output_width;

            // Do GEMV (note: this is a bit confusing because gemv assumes
            // column-major matrices)
            at::cuda::blas::gemv<scalar_t>(
                't',
                k_,
                m_,
                scale,
                grad_output_n.data_ptr<scalar_t>(),
                k_,
                ones.data_ptr<scalar_t>(),
                1,
                static_cast<scalar_t>(1),
                grad_bias.data_ptr<scalar_t>(),
                1);
          }
        }

        // Resize
        if (is_batch) {
          grad_output.resize_(
              {n_output_plane, output_depth, output_height, output_width});
          input.resize_(
              {input.size(1), input_depth, input_height, input_width});
        }
      });
}

} // namespace

Tensor& slow_conv_transpose3d_out_cuda(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  Tensor finput = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor fgrad = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  slow_conv_transpose3d_out_cuda_template(
      output,
      input,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      finput,
      fgrad);

  return output;
}

Tensor slow_conv_transpose3d_cuda(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor finput = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor fgrad = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  slow_conv_transpose3d_out_cuda_template(
      output,
      input,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      finput,
      fgrad);

  return output;
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv_transpose3d_backward_out_cuda(
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    const Tensor& finput,
    const Tensor& fgrad) {
  if (grad_input.defined()) {
    slow_conv_transpose3d_backward_out_cuda_template(
        input,
        grad_output,
        grad_input,
        weight,
        finput,
        fgrad,
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
    slow_conv_transpose3d_acc_grad_parameters_cuda(
        input,
        grad_output,
        grad_weight,
        grad_bias,
        finput,
        fgrad,
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

std::tuple<Tensor, Tensor, Tensor> slow_conv_transpose3d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    const Tensor& finput,
    const Tensor& fgrad,
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
    slow_conv_transpose3d_backward_out_cuda_template(
        input,
        grad_output,
        grad_input,
        weight,
        finput,
        fgrad,
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
    slow_conv_transpose3d_acc_grad_parameters_cuda(
        input,
        grad_output,
        grad_weight,
        grad_bias,
        finput,
        fgrad,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        1);
  }

  return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}

} // namespace native
} // namespace at

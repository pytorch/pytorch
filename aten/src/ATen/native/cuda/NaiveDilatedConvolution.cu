#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/vol2col.cuh>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/cuda/im2col.cuh>
#include <ATen/native/DilatedConvolutionUtils.h>
#include <c10/util/accumulate.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/slow_conv_dilated2d_native.h>
#include <ATen/ops/slow_conv_dilated3d_native.h>
#endif

#include <tuple>

namespace at::native {

namespace {

// hyper-volume to column, CUDA
template <typename Dtype, int64_t dim>
void hvol2col(
    cudaStream_t stream,
    const Dtype* data_hvol,
    const int channels,
    const IntArrayRef input_size,
    const IntArrayRef output_size,
    const IntArrayRef kernel_size,
    const IntArrayRef stride_size,
    const IntArrayRef pad_size,
    const IntArrayRef dilation_size,
    Dtype* data_col) {
  if (dim == 3) {
    vol2col<Dtype>(
        stream,
        data_hvol,
        channels,
        input_size[0],
        input_size[1],
        input_size[2],
        output_size[0],
        output_size[1],
        output_size[2],
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
        pad_size[0],
        pad_size[1],
        pad_size[2],
        stride_size[0],
        stride_size[1],
        stride_size[2],
        dilation_size[0],
        dilation_size[1],
        dilation_size[2],
        data_col);
  }
  if (dim == 2) {
    im2col<Dtype>(
        stream,
        data_hvol,
        channels,
        input_size[0],
        input_size[1],
        output_size[0],
        output_size[1],
        kernel_size[0],
        kernel_size[1],
        pad_size[0],
        pad_size[1],
        stride_size[0],
        stride_size[1],
        dilation_size[0],
        dilation_size[1],
        data_col);
  }
}

// column to hyper-volume, CUDA
template <typename Dtype, int64_t dim>
void col2hvol(
    cudaStream_t stream,
    const Dtype* data_col,
    const int channels,
    const IntArrayRef input_size,
    const IntArrayRef output_size,
    const IntArrayRef kernel_size,
    const IntArrayRef stride_size,
    const IntArrayRef pad_size,
    const IntArrayRef dilation_size,
    Dtype* data_hvol) {
  if (dim == 3) {
    col2vol<Dtype, Dtype>(
        stream,
        data_col,
        channels,
        input_size[0],
        input_size[1],
        input_size[2],
        output_size[0],
        output_size[1],
        output_size[2],
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
        pad_size[0],
        pad_size[1],
        pad_size[2],
        stride_size[0],
        stride_size[1],
        stride_size[2],
        dilation_size[0],
        dilation_size[1],
        dilation_size[2],
        data_hvol);
  }
  if (dim == 2) {
    col2im<Dtype, Dtype>(
        stream,
        data_col,
        channels,
        input_size[0],
        input_size[1],
        output_size[0],
        output_size[1],
        kernel_size[0],
        kernel_size[1],
        pad_size[0],
        pad_size[1],
        stride_size[0],
        stride_size[1],
        dilation_size[0],
        dilation_size[1],
        data_hvol);
  }
}

/*
   check tensor data locations
*/
void slow_conv_dilated_location_check(
    CheckedFrom c,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output) {
  // checking data locations of user-provided tensor arguments
  TensorArg input_arg{input, "input", 2}, weight_arg{weight, "weight", 3},
      bias_arg{bias, "bias", 4}, grad_output_arg{grad_output, "grad_output", 5};
  checkAllSameGPU(c, {input_arg, weight_arg});
  if (bias.defined()) {
    checkAllSameGPU(c, {input_arg, bias_arg});
  }
  if (grad_output.defined()) {
    checkAllSameGPU(c, {input_arg, grad_output_arg});
  }
  // we are not checking the data locations of other tensor
  // arguments such as output, grad_input, etc because of these are
  // allocated based on input options and hence these tensors always
  // have the same data location as of input tensor.
}

/*
  slow_conv_dilated_all_cuda_template

  Main worker. Computes tensors output, grad_input, grad_weight,
  and/or grad_bias if defined, respectively.
 */

template <int64_t dim>
void slow_conv_dilated_all_cuda_template(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  slow_conv_dilated_location_check(__func__, input, weight, bias, grad_output);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto options = input.options();
  // The rear part of input tensor sizes:
  auto input_size = input.sizes().slice(2);
  // The rear part of output tensor sizes:
  auto output_size = internal::get_output_size<dim>(
      input, kernel_size, stride_size, pad_size, dilation_size);
  int64_t batchSize = input.size(0);
  int64_t nInputPlane = weight.size(1);
  int64_t nOutputPlane = weight.size(0);
  // Temporary buffers:
  const int64_t m = c10::multiply_integers(kernel_size);
  const int64_t output_vsize = c10::multiply_integers(output_size);
  Tensor columns = at::empty({0}, options);
  if (output.defined() || grad_weight.defined() || grad_input.defined()) {
    columns.resize_({nInputPlane * m, output_vsize});
  }
  // Initialize
  if (grad_weight.defined()) {
    grad_weight.zero_();
  }
  if (grad_bias.defined()) {
    grad_bias.zero_();
  }
  if (output.defined() && !bias.defined()) {
    output.zero_();
  }

#if defined(USE_ROCM)
  /* When using ROCm, the sum evaluation is inaccurate for double
     tensors. The reason is currently unknown. Hence, we use gemv for
     computing `grad_output_n.sum(dims)` until the ROCm-sum issue is
     resolved. */
  Tensor ones = at::empty({0}, options);
  if (grad_bias.defined()) {
    ones.resize_({output_vsize});
    ones.fill_(1);
  }
  /* MSVC does not like #ifdef-s inside the CPP macro
     AT_DISPATCH_FLOATING_TYPES_AND_HALF. So, we define the code
     branching outside the CPP macro: */
#define CALCULATE_GRAD_BIAS                                \
  at::cuda::blas::gemv<scalar_t>(                          \
      /*trans=*/'t',                                       \
      /*    m=*/output_vsize,                              \
      /*    n=*/nOutputPlane,                              \
      /*alpha=*/static_cast<scalar_t>(1),                  \
      /*    A=*/grad_output_n.const_data_ptr<scalar_t>(),  \
      /*  lda=*/output_vsize,                              \
      /*    x=*/ones.const_data_ptr<scalar_t>(),           \
      /* incx=*/1,                                         \
      /* beta=*/static_cast<scalar_t>(1),                  \
      /*    y=*/grad_bias.mutable_data_ptr<scalar_t>(),    \
      /* incy=*/1)
#else
#define CALCULATE_GRAD_BIAS grad_bias += grad_output_n.sum(dims)
#endif

  // Helpers
  Tensor grad_output_n;
  std::vector<int64_t> dims(dim);
  std::iota(dims.begin(), dims.end(), 1);

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
      input.scalar_type(), "slow_conv_dilated<>", [&] {
        // For each elt in batch, do:
        for (int elt = 0; elt < batchSize; elt++) {
          // Matrix multiply per output:
          Tensor input_n = input.select(0, elt);

          // Output
          if (output.defined()) {
            Tensor output_n = output.select(0, elt);
            if (bias.defined()) {
              /* For gemm argument derivation, see
                 slow_conv_dilated_all_cuda_template in
                 ATen/native/DilatedConvolution.cpp */
              for (int n = 0; n < nOutputPlane; n++) {
                output_n.select(0, n).fill_(bias[n]);
              }
            }
            // Extract columns:
            hvol2col<scalar_t, dim>(
                stream,
                input_n.const_data_ptr<scalar_t>(),
                nInputPlane,
                input_size,
                output_size,
                kernel_size,
                stride_size,
                pad_size,
                dilation_size,
                columns.mutable_data_ptr<scalar_t>());
            /* For gemm argument derivation, see
               slow_conv_dilated_all_cuda_template in
               ATen/native/DilatedConvolution.cpp */
            at::cuda::blas::gemm<scalar_t>(
                /*transa=*/'n',
                /*transb=*/'n',
                /*     m=*/columns.size(1),
                /*     n=*/nOutputPlane,
                /*     k=*/columns.size(0),
                /* alpha=*/static_cast<scalar_t>(1),
                /*     A=*/columns.const_data_ptr<scalar_t>(),
                /*   lda=*/columns.size(1),
                /*     B=*/weight.const_data_ptr<scalar_t>(),
                /*   ldb=*/columns.size(0),
                /*  beta=*/static_cast<scalar_t>(1),
                /*     C=*/output_n.mutable_data_ptr<scalar_t>(),
                /*   ldc=*/columns.size(1));

          } else {
            // All gradients
            grad_output_n = grad_output.select(0, elt);
          }

          // Gradient of input:
          if (grad_input.defined()) {
            /* For gemm argument derivation, see
               slow_conv_dilated_all_cuda_template in
               ATen/native/DilatedConvolution.cpp */
            at::cuda::blas::gemm<scalar_t>(
                /*transa=*/'n',
                /*transb=*/'t',
                /*     m=*/columns.size(1),
                /*     n=*/columns.size(0),
                /*     k=*/nOutputPlane,
                /* alpha=*/static_cast<scalar_t>(1),
                /*     A=*/grad_output_n.const_data_ptr<scalar_t>(),
                /*   lda=*/columns.size(1),
                /*     B=*/weight.const_data_ptr<scalar_t>(),
                /*   ldb=*/columns.size(0),
                /*  beta=*/static_cast<scalar_t>(0),
                /*     C=*/columns.mutable_data_ptr<scalar_t>(),
                /*   ldc=*/columns.size(1));
            // Unpack columns back into input:
            Tensor grad_input_n = grad_input.select(0, elt);

            col2hvol<scalar_t, dim>(
                stream,
                columns.const_data_ptr<scalar_t>(),
                nInputPlane,
                input_size,
                output_size,
                kernel_size,
                stride_size,
                pad_size,
                dilation_size,
                grad_input_n.mutable_data_ptr<scalar_t>());
          }

          // Gradient of weight:
          if (grad_weight.defined()) {
            // Extract columns:
            hvol2col<scalar_t, dim>(
                stream,
                input_n.const_data_ptr<scalar_t>(),
                nInputPlane,
                input_size,
                output_size,
                kernel_size,
                stride_size,
                pad_size,
                dilation_size,
                columns.mutable_data_ptr<scalar_t>());
            scalar_t scale = static_cast<scalar_t>(
                1); // TODO: expose as argument?
            /* For gemm argument derivation, see
               slow_conv_dilated_all_cuda_template in
               ATen/native/DilatedConvolution.cpp */
            at::cuda::blas::gemm<scalar_t>(
                /*transa=*/'t',
                /*transb=*/'n',
                /*     m=*/columns.size(0),
                /*     n=*/nOutputPlane,
                /*     k=*/columns.size(1),
                /* alpha=*/scale,
                /*     A=*/columns.const_data_ptr<scalar_t>(),
                /*   lda=*/columns.size(1),
                /*     B=*/grad_output_n.const_data_ptr<scalar_t>(),
                /*   ldb=*/columns.size(1),
                /*  beta=*/static_cast<scalar_t>(1),
                /*     C=*/grad_weight.mutable_data_ptr<scalar_t>(),
                /*   ldc=*/columns.size(0));
          }

          // Gradient of bias:
          if (grad_bias.defined()) {
            /* For gemv argument derivation, see
               slow_conv_dilated_all_cpu_template in
               ATen/native/DilatedConvolution.cpp */
            CALCULATE_GRAD_BIAS; /* MSVC does not like #ifdef-s
                                    inside the CPP macros, see above. */
            /*
              TODO: when scale != 1 is introduced then use:
                grad_bias += scale * grad_output_n.sum(dims);
             */
          }
        }
      });

} // slow_conv_dilated_all_cuda_template

} // namespace

Tensor slow_conv_dilated2d_cuda(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  Tensor undefined;
  internal::slow_conv_dilated_shape_check<2>(
      input,
      weight,
      bias,
      undefined,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  auto is_batch = input.dim() == 4;
  auto options = input.options();
  // calculate output tensor size
  auto output_size = internal::get_output_size<2>(
      input, weight, kernel_size, stride_size, pad_size, dilation_size);
  // template function assumes batched tensors.  unsqueeze(0) will
  // insert batch dimension without affecting the original tensor.
  const Tensor input_ =
      (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
  const Tensor weight_ = weight.contiguous();
  const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);
  Tensor output = at::empty(output_size, options);
  Tensor output_ = (is_batch ? output : output.unsqueeze(0));

  slow_conv_dilated_all_cuda_template<2>(
      output_,
      input_,
      weight_,
      bias_,
      undefined,
      undefined,
      undefined,
      undefined,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return output;
}

std::tuple<Tensor, Tensor, Tensor> slow_conv_dilated2d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    const std::array<bool, 3ul> output_mask) {
  Tensor undefined;
  internal::slow_conv_dilated_shape_check<2>(
      input,
      weight,
      undefined,
      grad_output,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  auto is_batch = input.dim() == 4;
  auto options = grad_output.options();
  // template function assumes batched tensors.  unsqueeze(0) will
  // insert batch dimension without affecting the original tensor.
  const Tensor grad_output_ =
      (is_batch ? grad_output.contiguous()
                : grad_output.contiguous().unsqueeze(0));
  const Tensor input_ =
      (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
  const Tensor weight_ = weight.contiguous();
  // compute only gradients for which the corresponding output_mask is true:
  Tensor grad_input =
      (output_mask[0] ? at::empty(input.sizes(), options) : undefined);
  Tensor grad_weight =
      (output_mask[1] ? at::empty(weight.sizes(), options) : undefined);
  Tensor grad_bias =
      (output_mask[2] ? at::empty(weight.size(0), options) : undefined);
  Tensor grad_input_ =
      (output_mask[0] ? (is_batch ? grad_input : grad_input.unsqueeze(0))
                      : undefined);
  slow_conv_dilated_all_cuda_template<2>(
      undefined,
      input_,
      weight_,
      undefined,
      grad_output_,
      grad_input,
      grad_weight,
      grad_bias,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return std::tie(grad_input, grad_weight, grad_bias);
}

Tensor slow_conv_dilated3d_cuda(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  Tensor undefined;
  internal::slow_conv_dilated_shape_check<3>(
      input,
      weight,
      bias,
      undefined,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  auto is_batch = input.dim() == 5;
  auto options = input.options();
  // calculate output tensor size
  auto output_size = internal::get_output_size<3>(
      input, weight, kernel_size, stride_size, pad_size, dilation_size);
  // template function assumes batched tensors.  unsqueeze(0) will
  // insert batch dimension without affecting the original tensor.
  const Tensor input_ =
      (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
  const Tensor weight_ = weight.contiguous();
  const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);
  Tensor output = at::empty(output_size, options);
  Tensor output_ = (is_batch ? output : output.unsqueeze(0));

  slow_conv_dilated_all_cuda_template<3>(
      output,
      input_,
      weight_,
      bias_,
      undefined,
      undefined,
      undefined,
      undefined,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return output;
}

std::tuple<Tensor, Tensor, Tensor> slow_conv_dilated3d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    const std::array<bool, 3ul> output_mask) {
  Tensor undefined;
  internal::slow_conv_dilated_shape_check<3>(
      input,
      weight,
      undefined,
      grad_output,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  auto is_batch = input.dim() == 5;
  auto options = grad_output.options();
  // template function assumes batched tensors.  unsqueeze(0) will
  // insert batch dimension without affecting the original tensor.
  const Tensor grad_output_ =
      (is_batch ? grad_output.contiguous()
                : grad_output.contiguous().unsqueeze(0));
  const Tensor input_ =
      (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
  const Tensor weight_ = weight.contiguous();
  // compute only gradients for which the corresponding output_mask is true:
  Tensor grad_input =
      (output_mask[0] ? at::empty(input.sizes(), options) : undefined);
  Tensor grad_weight =
      (output_mask[1] ? at::empty(weight.sizes(), options) : undefined);
  Tensor grad_bias =
      (output_mask[2] ? at::empty(weight.size(0), options) : undefined);
  Tensor grad_input_ =
      (output_mask[0] ? (is_batch ? grad_input : grad_input.unsqueeze(0))
                      : undefined);
  slow_conv_dilated_all_cuda_template<3>(
      undefined,
      input_,
      weight_,
      undefined,
      grad_output_,
      grad_input,
      grad_weight,
      grad_bias,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return std::tie(grad_input, grad_weight, grad_bias);
}

REGISTER_CUDA_DISPATCH(slow_conv_dilated2d_backward_stub, &slow_conv_dilated2d_backward_cuda);
REGISTER_CUDA_DISPATCH(slow_conv_dilated3d_backward_stub, &slow_conv_dilated3d_backward_cuda);

} // namespace at::native

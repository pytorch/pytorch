#include <ATen/ATen.h>

#include <ATen/div_rtn.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/im2col.cuh>

namespace at { namespace native {
namespace {

void slow_conv2d_shape_check(
    const Tensor& input, const Tensor& grad_output,
    const Tensor& weight, const Tensor& bias,
    int64_t kH, int64_t kW,
    int64_t dH, int64_t dW,
    int64_t padH, int64_t padW,
    bool weight_nullable) {
  TORCH_CHECK(kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got kH: ", kH, " kW: ", kW);
  TORCH_CHECK(dW > 0 && dH > 0,
              "stride should be greater than zero, but got dH: ", dH, " dW: ", dW);

  TORCH_CHECK(weight_nullable || weight.defined(),
              "weight tensor is expected to be non-nullable");
  TORCH_CHECK(!weight.defined() ||
              ((weight.numel() > 0) && (weight.dim() == 2 || weight.dim() == 4)),
              "non-empty 2D or 4D weight tensor expected, but got: ", weight.sizes());
  TORCH_CHECK(!bias.defined() || (bias.dim() == 1 && bias.sizes()[0] == weight.sizes()[0]),
              "Expected bias to have shape [", weight.sizes()[0], "] but got ", bias.sizes());

  const auto in_sizes = input.sizes();
  int ndim = in_sizes.size();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  // Allow for empty batch size but not other dimensions
  const bool valid_empty = c10::multiply_integers(in_sizes.slice(1)) != 0;
  TORCH_CHECK((valid_empty) && (ndim == 3 || ndim == 4),
              "non-empty 3D or 4D input tensor expected but got: ", in_sizes);

  int64_t inputHeight = in_sizes[dimh];
  int64_t inputWidth = in_sizes[dimw];

  int64_t exactInputHeight = inputHeight + 2 * padH;
  int64_t exactInputWidth = inputWidth + 2 * padW;

  TORCH_CHECK(exactInputHeight >= kH && exactInputWidth >= kW,
              "Calculated padded input size per channel: ",
              IntArrayRef{exactInputHeight, exactInputWidth},
              ". Kernel size: ", IntArrayRef{kH, kW},
              ". Kernel size can't be greater than actual input size");

  // NOTE: can't use conv_output_size if the weight isn't defined
  auto outputHeight = div_rtn<int64_t>(exactInputHeight - kH, dH) + 1;
  auto outputWidth = div_rtn<int64_t>(exactInputWidth - kW, dW) + 1;

  TORCH_CHECK(outputWidth >= 1 && outputHeight >= 1,
              "Given input size per channel: ",
              IntArrayRef{inputHeight, inputWidth},
              ". Calculated output size per channel: ",
              IntArrayRef{outputHeight, outputWidth},
              ". Output size is too small");

  if (weight.defined()) {
    const auto w_sizes = weight.sizes();
    int64_t nInputPlane = w_sizes[1];
    if (w_sizes.size() == 2) {
      nInputPlane /= (kH * kW);
    }
    TORCH_CHECK(in_sizes[dimf] == nInputPlane,
                "Expected input dim ", dimf, " to have size ", nInputPlane,
                " but got ", in_sizes[dimf]);
  }

  if (grad_output.defined()) {
    const auto gO_sizes = grad_output.sizes();
    TORCH_CHECK(gO_sizes.size() == ndim,
                "Expected grad_output to have ", ndim,
                " dimensions but got shape", gO_sizes);

    if (weight.defined()) {
      const auto w_sizes = weight.sizes();
      TORCH_CHECK(gO_sizes[dimf] == w_sizes[0],
                  "Expected  dim ", dimf, " to have size ", w_sizes[0],
                  " but got ", gO_sizes[dimf]);
    } else if (bias.defined()) {
      const auto b_sizes = bias.sizes();
      int64_t nOutputPlane = b_sizes.size() == 0 ? 1 : b_sizes[0];
      TORCH_CHECK(gO_sizes[dimf] == nOutputPlane,
                  "Expected grad_output dim ", dimf, " to have size ",
                  nOutputPlane, " but got ", gO_sizes[dimf]);
    }
    TORCH_CHECK(gO_sizes[dimh] == outputHeight,
                "Expected grad_output dim ", dimh, " to have size ",
                outputHeight, " but got ", gO_sizes[dimh]);
    TORCH_CHECK(gO_sizes[dimw] == outputWidth,
                "Expected grad_output dim ", dimw, " to have size ",
                outputWidth, " but got ", gO_sizes[dimw]);
  }
}

Tensor new_view_weight_MM2d(const Tensor& weight_) {
  auto weight = weight_.expect_contiguous();
  if (weight->dim() != 4) {
    return *weight;
  }
  const auto w_sizes = weight->sizes();
  int64_t s1 = w_sizes[0];
  int64_t s2 = c10::multiply_integers(w_sizes.slice(1));
  return weight->view({s1, s2});
}

void slow_conv2d_forward(
           const Tensor &input_,
           const Tensor &output_,
           const Tensor &weight_,
           const Tensor &bias,
           const Tensor &columns,
           const Tensor &ones_,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW) {
  auto weight = new_view_weight_MM2d(weight_);
  slow_conv2d_shape_check(
      input_, {}, weight, bias, kH, kW, dH, dW, padH, padW, /*weight_nullable*/false);

  TORCH_CHECK(!bias.defined() || bias.is_contiguous(),
              "bias tensor has to be contiguous");

  int ndim = input_.dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  auto in_sizes = input_.sizes();
  int64_t nInputPlane  = in_sizes[dimf];
  int64_t inputHeight  = in_sizes[dimh];
  int64_t inputWidth   = in_sizes[dimw];
  int64_t nOutputPlane = weight.sizes()[0];
  int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

  Tensor input = input_;
  if (in_sizes.size() == 3) {
    input = input.unsqueeze(0);
  }

  // Batch size + input planes
  in_sizes = input.sizes();
  int64_t batchSize = in_sizes[0];

  // Resize output
  Tensor output = output_;
  if (ndim == 4) {
    resize_output(output, {batchSize, nOutputPlane, outputHeight, outputWidth});
  } else {
    resize_output(output, {nOutputPlane, outputHeight, outputWidth});
    output = output.unsqueeze(0);
  }

  // Resize temporary columns
  resize_output(columns, {nInputPlane * kW * kH, outputHeight * outputWidth});

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  Tensor ones;
  if (bias.defined()) {
    ones = at::ones({outputHeight, outputWidth}, input.options());
  }
  const bool requires_columns = (
      kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                  "slow_conv2d_cuda", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt ++) {
      // Matrix mulitply per output:
      auto input_n = input.select(0, elt);
      auto output_n = output.select(0, elt);

      // Do Bias first:
      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m_ = nOutputPlane;
      int64_t n_ = outputHeight * outputWidth;
      int64_t k_ = 1;

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      if (bias.defined()) {
        at::cuda::blas::gemm(
            't', 'n',
            n_, m_, k_,
            scalar_t(1),
            ones.data_ptr<scalar_t>(), k_,
            bias.data_ptr<scalar_t>(), k_,
            scalar_t(0),
            output_n.data_ptr<scalar_t>(), n_
        );
      } else {
        output_n.zero_();
      }

      if (requires_columns) {
        // Extract columns:
        at::native::im2col(
          c10::cuda::getCurrentCUDAStream(),
          input_n.data_ptr<scalar_t>(),
          nInputPlane, inputHeight, inputWidth,
          outputHeight, outputWidth,
          kH, kW, padH, padW, dH, dW,
          1, 1,
          columns.data_ptr<scalar_t>()
        );
      }

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m = nOutputPlane;
      int64_t n = columns.size(1);
      int64_t k = nInputPlane*kH*kW;

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      auto gemm_in_ptr = requires_columns ?
          columns.data_ptr<scalar_t>() :
          input_n.data_ptr<scalar_t>();
      at::cuda::blas::gemm(
          'n', 'n',
          n, m, k,
          scalar_t(1),
          gemm_in_ptr, n,
          weight.data_ptr<scalar_t>(), k,
          scalar_t(1),
          output_n.data_ptr<scalar_t>(), n
      );
    }
  });
}

void slow_conv2d_backward(
    const Tensor &input_,
    const Tensor &grad_output_,
    const Tensor &grad_input_,
    const Tensor &weight_,
    const Tensor &grad_columns,
    const Tensor &ones,
    int kH, int kW,
    int dH, int dW,
    int padH, int padW) {
  Tensor weight = new_view_weight_MM2d(weight_);
  slow_conv2d_shape_check(input_, grad_output_, weight, {},
                          kH, kW, dH, dW, padH, padW, /*weight_nullable=*/false);

  // Params
  auto weight_sizes = weight.sizes();
  int nInputPlane = weight_sizes.size() == 2 ? weight_sizes[1]/(kW*kH) : weight_sizes[1];
  int nOutputPlane = weight_sizes[0];

  TORCH_INTERNAL_ASSERT(input_.is_contiguous());
  TORCH_INTERNAL_ASSERT(grad_output_.is_contiguous());

  Tensor input = input_;
  Tensor grad_output = grad_output_;
  const bool is_batched = input_.dim() == 4;
  if (!is_batched) {
    // Force batch
    input = input_.unsqueeze(0);
    grad_output = grad_output_.unsqueeze(0);
  }

  auto input_sizes = input.sizes();
  int64_t inputWidth   = input_sizes[3];
  int64_t inputHeight  = input_sizes[2];
  auto output_sizes = grad_output.sizes();
  int64_t outputWidth  = output_sizes[3];
  int64_t outputHeight = output_sizes[2];

  // Batch size + input planes
  int64_t batchSize = input_sizes[0];

  // Resize output
  resize_output(grad_input_, input_.sizes());
  TORCH_CHECK(grad_input_.is_contiguous(), "grad_input must be contiguous");
  const auto grad_input = is_batched ? grad_input_ : grad_input_.unsqueeze(0);

  // Resize temporary columns
  resize_output(grad_columns, {nInputPlane*kW*kH, outputHeight*outputWidth});
  TORCH_CHECK(grad_columns.is_contiguous(), "grad_columns must be contiguous");

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                  "slow_conv2d_backward_cuda", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt ++) {
      // Matrix mulitply per sample:
      auto grad_input_n = grad_input.select(0, elt);
      auto grad_output_n = grad_output.select(0, elt);

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m = nInputPlane*kW*kH;
      int64_t n = grad_columns.sizes()[1];
      int64_t k = nOutputPlane;

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      at::cuda::blas::gemm<scalar_t>(
          'n', 't',
          n, m, k,
          scalar_t(1),
          grad_output_n.data_ptr<scalar_t>(), n,
          weight.data_ptr<scalar_t>(), m,
          scalar_t(0),
          grad_columns.data_ptr<scalar_t>(), n
      );

      // Unpack columns back into input:
      using acc_t = at::acc_type<scalar_t, true>;
      at::native::col2im<scalar_t, acc_t>(
        c10::cuda::getCurrentCUDAStream(),
        grad_columns.data_ptr<scalar_t>(),
        nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
        1, 1, grad_input_n.data_ptr<scalar_t>()
      );
    }
  });
}

void slow_conv2d_grad_weight_bias(
           const Tensor &input_,
           const Tensor &grad_output_,
           const Tensor &grad_weight_,
           const Tensor &grad_bias,
           const Tensor &columns,
           const Tensor &ones,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW) {
  if (grad_weight_.defined()) {
    TORCH_CHECK(grad_weight_.is_contiguous(), "grad_weight needs to be contiguous");
  }
  if (grad_bias.defined()) {
    TORCH_CHECK(grad_bias.is_contiguous(), "grad_bias needs to be contiguous");
    TORCH_CHECK(ones.is_contiguous(), "ones needs to be contiguous");
  }

  auto grad_weight = new_view_weight_MM2d(grad_weight_);
  slow_conv2d_shape_check(input_, grad_output_, grad_weight, grad_bias,
                          kH, kW, dH, dW, padH, padW, /*weight_nullable=*/true);

  // Params
  TORCH_INTERNAL_ASSERT(input_.is_contiguous());
  TORCH_INTERNAL_ASSERT(grad_output_.is_contiguous());

  Tensor input = input_;
  Tensor grad_output = grad_output_;
  if (input.dim() == 3) {
    input = input_.unsqueeze(0);
    grad_output = grad_output_.unsqueeze(0);
  }

  auto input_sizes = input.sizes();
  int64_t nInputPlane = input_sizes[1];
  int64_t nOutputPlane = grad_output.sizes()[1];

  int64_t inputWidth   = input_sizes[3];
  int64_t inputHeight  = input_sizes[2];
  int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input_sizes[0];

  // Define a buffer of ones, for bias accumulation
  if (ones.defined() && ones.numel() < outputHeight * outputWidth) {
    ones.resize_({outputHeight, outputWidth});
    ones.fill_(1);
  }

  // Resize temporary columns
  resize_output(columns, {nInputPlane * kH * kW, outputHeight * outputWidth});

  const bool requires_columns = (
      kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                  "slow_conv2d_grad_weight_bias_cuda", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt ++) {
      // Matrix mulitply per output:
      auto grad_output_n = grad_output.select(0, elt);

      // Do Weight:
      if (grad_weight.defined()) {
        // Matrix mulitply per output:
        auto input_n = input.select(0, elt);

        if (requires_columns) {
          // Extract columns:
          at::native::im2col<scalar_t>(
            c10::cuda::getCurrentCUDAStream(),
            input_n.data_ptr<scalar_t>(),
            nInputPlane, inputHeight, inputWidth,
            outputHeight, outputWidth,
            kH, kW, padH, padW, dH, dW,
            1, 1,
            columns.data_ptr<scalar_t>()
          );
        }

        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        int64_t m = nOutputPlane;
        int64_t n = nInputPlane*kW*kH;
        int64_t k = columns.sizes()[1];

        // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
        auto gemm_in_ptr = requires_columns ?
            columns.data_ptr<scalar_t>() :
            input_n.data_ptr<scalar_t>();
        at::cuda::blas::gemm(
            't', 'n',
            n, m, k,
            scalar_t(1),
            gemm_in_ptr, k,
            grad_output_n.data_ptr<scalar_t>(), k,
            scalar_t(1),
            grad_weight.data_ptr<scalar_t>(), n
        );
      }

      // Do Bias:
      if (grad_bias.defined()) {
        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        int64_t m_ = nOutputPlane;
        int64_t k_ = outputHeight * outputWidth;

        // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
        at::cuda::blas::gemv(
            't',
            k_, m_,
            scalar_t(1),
            grad_output_n.data_ptr<scalar_t>(), k_,
            ones.data_ptr<scalar_t>(), 1,
            scalar_t(1),
            grad_bias.data_ptr<scalar_t>(), 1
        );
      }
    }
  });
}

}  // namespace (anonymous)


std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_forward_out_cuda(
    const Tensor &self_,
    const Tensor &weight_,
    IntArrayRef kernel_size,
    const c10::optional<Tensor> &bias_,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor &output,
    Tensor &finput,
    Tensor &fgrad_input) {
  TORCH_CHECK(kernel_size.size() == 2);
  TORCH_CHECK(stride.size() == 2);
  TORCH_CHECK(padding.size() == 2);

  auto self = self_.expect_contiguous();
  auto weight = weight_.expect_contiguous();
  auto bias = [&] {
    if (bias_.has_value() && bias_->defined()) {
      return bias_->expect_contiguous();
    }
    return MaybeOwned<Tensor>::owned(c10::in_place);
  }();

  slow_conv2d_forward(
      *self,
      output,
      *weight,
      *bias,
      finput,
      fgrad_input,
      kernel_size[0], kernel_size[1],
      stride[0], stride[1],
      padding[0], padding[1]
    );
  return std::tuple<Tensor&, Tensor&, Tensor&>{
      output, finput, fgrad_input};
}

std::tuple<Tensor, Tensor, Tensor> slow_conv2d_forward_cuda(
    const Tensor &self,
    const Tensor &weight,
    IntArrayRef kernel_size,
    const c10::optional<Tensor> &bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto output = at::empty({0}, self.options());
  auto finput = at::empty({0}, self.options());
  auto fgrad_input = at::empty({0}, self.options());
  return slow_conv2d_forward_out_cuda(
      self, weight, kernel_size, bias, stride, padding, output, finput, fgrad_input);
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_backward_out_cuda(const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const Tensor& finput,
    const Tensor& fgrad_input,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  if (grad_weight.defined()) {
    resize_output(grad_weight, weight_.sizes());
    grad_weight.zero_();
  }
  if (grad_bias.defined()) {
    resize_output(grad_bias, {weight_.sizes()[0]});
    grad_bias.zero_();
  }
  if (grad_input.defined()) {
    resize_output(grad_input, self.sizes());
    auto weight = weight_.expect_contiguous();

    slow_conv2d_backward(
        self, grad_output,
        grad_input, *weight,
        finput, fgrad_input,
        kernel_size[0], kernel_size[1],
        stride[0], stride[1],
        padding[0], padding[1]);
  }
  if (grad_weight.defined() || grad_bias.defined()) {
    slow_conv2d_grad_weight_bias(
        self,
        grad_output,
        grad_weight,
        grad_bias,
        finput,
        fgrad_input,
        kernel_size[0], kernel_size[1],
        stride[0], stride[1],
        padding[0], padding[1]
      );
  }
  return std::tuple<Tensor&, Tensor&, Tensor&>{
      grad_input, grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> slow_conv2d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const Tensor& finput,
    const Tensor& fgrad_input,
    std::array<bool, 3> output_mask) {
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  }

  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  }

  if (output_mask[2]) {
    grad_bias = at::empty({0}, grad_output.options());
  }

  return native::slow_conv2d_backward_out_cuda(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      finput,
      fgrad_input,
      grad_input,
      grad_weight,
      grad_bias);
}

} // namespace native
} // namespace at

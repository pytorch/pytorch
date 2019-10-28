#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/div_rtn.h>

namespace at {
namespace native {

namespace {

static inline void conv2d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& bias,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    bool weight_nullable) {
  TORCH_CHECK(
      kW > 0 && kH > 0,
      "kernel size should be greater than zero, but got kH: ",
      kH,
      " kW: ",
      kW);
  TORCH_CHECK(
      dW > 0 && dH > 0,
      "stride should be greater than zero, but got dH: ",
      dH,
      " dW: ",
      dW);

  if (weight.defined()) {
    TORCH_CHECK(
        !weight.numel() == 0 && (weight.dim() == 2 || weight.dim() == 4),
        "non-empty 2D or 4D weight tensor expected, but got: ",
        weight.sizes());
    if (bias.defined()) {
      check_dim_size(bias, 1, 0, weight.size(0));
    }
  } else {
    TORCH_CHECK(!weight_nullable, "weight tensor is undefined");
  }

  const int64_t ndim = input.dim();
  int64_t dimf = 0;
  int64_t dimh = 1;
  int64_t dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  // Allow for empty batch size but not other dimensions
  bool valid_empty = false;
  if (ndim == 3) {
    valid_empty =
        input.size(0) == 0 && input.size(1) != 0 && input.size(2) != 0;
  } else if (ndim == 4) {
    valid_empty = input.size(0) == 0 && input.size(1) != 0 &&
        input.size(2) != 0 && input.size(3) != 0;
  }

  TORCH_CHECK(
      (input.numel() > 0 || valid_empty) && (ndim == 3 || ndim == 4),
      "non-empty 3D or 4D input tensor expected but got: ",
      input.sizes());

  const int64_t inputHeight = input.size(dimh);
  const int64_t inputWidth = input.size(dimw);

  const int64_t exactInputHeight = inputHeight + 2 * padH;
  const int64_t exactInputWidth = inputWidth + 2 * padW;

  TORCH_CHECK(
      exactInputHeight >= kH && exactInputWidth >= kW,
      "Calculated padded input size per channel: (",
      exactInputHeight,
      " x ",
      exactInputWidth,
      "). ",
      "Kernel size: (",
      kH,
      " x ",
      kW,
      "). Kernel size can't be greater than actual input size");

  int64_t outputHeight = div_rtn<int64_t>(exactInputHeight - kH, dH) + 1;
  int64_t outputWidth = div_rtn<int64_t>(exactInputWidth - kW, dW) + 1;

  TORCH_CHECK(
      outputWidth >= 1 && outputHeight >= 1,
      "Given input size per channel: (",
      inputHeight,
      " x ",
      inputWidth,
      "). "
      "Calculated output size per channel: (",
      outputHeight,
      " x ",
      outputWidth,
      "). Output size is too small");

  if (weight.defined()) {
    int64_t nInputPlane = weight.size(1);
    if (weight.dim() == 2) {
      nInputPlane /= (kH * kW);
    }
    check_dim_size(input, ndim, dimf, nInputPlane);
  }

  if (grad_output.defined()) {
    if (weight.defined()) {
      int64_t nOutputPlane = weight.size(0);
      check_dim_size(grad_output, ndim, dimf, nOutputPlane);
    } else if (bias.defined()) {
      TORCH_CHECK(bias.numel() > 0, "non-empty bias tensor expected");
      const int64_t nOutputPlane = bias.dim() == 0 ? 1 : bias.size(0);
      check_dim_size(grad_output, ndim, dimf, nOutputPlane);
    }
    check_dim_size(grad_output, ndim, dimh, outputHeight);
    check_dim_size(grad_output, ndim, dimw, outputWidth);
  }
}

/* note: due to write issues, this one cannot be parallelized as well as
 * unfolded_copy */
/*void unfolded_acc(
    Tensor finput,
    Tensor input,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int nInputPlane,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight) {
  // This function assumes that
  // outputHeight*dH does not overflow a int64_t
  // outputWidth*dW does not overflow a int64_t

  scalar_t* input_data = input->data<scalar_t>();
  scalar_t* finput_data = finput->data<scalar_t>();

  at::parallel_for(0, nInputPlane, 0, [&](int64_t start, int64_t end) {
    for (auto nip = start; nip < end; nip++) {
      int kw, kh, y, x;
      int64_t ix, iy;
      for (kh = 0; kh < kH; kh++) {
        for (kw = 0; kw < kW; kw++) {
          scalar_t* src = finput_data +
              nip * ((size_t)kH * kW * outputHeight * outputWidth) +
              kh * ((size_t)kW * outputHeight * outputWidth) +
              kw * ((size_t)outputHeight * outputWidth);
          scalar_t* dst = input_data + nip * ((size_t)inputHeight * inputWidth);
          if (padW > 0 || padH > 0) {
            int lpad, rpad;
            for (y = 0; y < outputHeight; y++) {
              iy = (int64_t)y * dH - padH + kh;
              if (iy < 0 || iy >= inputHeight) {
              } else {
                if (dW == 1) {
                  ix = 0 - padW + kw;
                  lpad = fmaxf(0, padW - kw);
                  rpad = fmaxf(0, padW - (kW - kw - 1));
                  scalar_t* dst_slice =
                      dst + (size_t)iy * inputWidth + ix + lpad;
                  THVector_(cadd)(
                      dst_slice,
                      dst_slice,
                      src + (size_t)y * outputWidth + lpad,
                      1,
                      outputWidth - lpad - rpad); // note: THVector_add could
                                                  //   handle 1 value better
                } else {
                  for (x = 0; x < outputWidth; x++) {
                    ix = (int64_t)x * dW - padW + kw;
                    if (ix < 0 || ix >= inputWidth) {
                    } else {
                      scalar_t* dst_slice = dst + (size_t)iy * inputWidth + ix;
                      THVector_(cadd)(
                          dst_slice,
                          dst_slice,
                          src + (size_t)y * outputWidth + x,
                          1,
                          1);
                    }
                  }
                }
              }
            }
          } else {
            for (y = 0; y < outputHeight; y++) {
              iy = (int64_t)y * dH + kh;
              ix = 0 + kw;
              if (dW == 1) {
                scalar_t* dst_slice = dst + (size_t)iy * inputWidth + ix;
                THVector_(cadd)(
                    dst_slice,
                    dst_slice,
                    src + (size_t)y * outputWidth,
                    1,
                    outputWidth); // note: THVector_add could handle 1 value
                                  // better
              } else {
                for (x = 0; x < outputWidth; x++) {
                  scalar_t* dst_slice =
                      dst + (size_t)iy * inputWidth + ix + x * dW;
                  THVector_(cadd)(
                      dst_slice,
                      dst_slice,
                      src + (size_t)y * outputWidth + x,
                      1,
                      1);
                }
              }
            }
          }
        }
      }
    }
  });
} */

template <typename scalar_t>
void unfolded_copy_kernel(
    scalar_t* input_data,
    scalar_t* finput_data,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int nInputPlane,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight) {
  at::parallel_for(
      0, (int64_t)nInputPlane * kH * kW, 0, [&](int64_t start, int64_t end) {
        for (auto k = start; k < end; k++) {
          int64_t nip = k / (kH * kW);
          int64_t rest = k % (kH * kW);
          int64_t kh = rest / kW;
          int64_t kw = rest % kW;
          int x, y;
          int64_t ix, iy;
          scalar_t* dst = finput_data +
              nip * ((size_t)kH * kW * outputHeight * outputWidth) +
              kh * ((size_t)kW * outputHeight * outputWidth) +
              kw * ((size_t)outputHeight * outputWidth);
          scalar_t* src = input_data + nip * ((size_t)inputHeight * inputWidth);
          if (padW > 0 || padH > 0) {
            int64_t lpad, rpad;
            for (y = 0; y < outputHeight; y++) {
              iy = (int64_t)y * dH - padH + kh;
              if (iy < 0 || iy >= inputHeight) {
                memset(
                    dst + (size_t)y * outputWidth,
                    0,
                    sizeof(scalar_t) * outputWidth);
              } else {
                if (dW == 1) {
                  ix = 0 - padW + kw;
                  lpad = fmaxf(0, padW - kw);
                  rpad = fmaxf(0, padW - (kW - kw - 1));
                  if (outputWidth - rpad - lpad <= 0) {
                    memset(
                        dst + (size_t)y * outputWidth,
                        0,
                        sizeof(scalar_t) * outputWidth);
                  } else {
                    if (lpad > 0)
                      memset(
                          dst + (size_t)y * outputWidth,
                          0,
                          sizeof(scalar_t) * lpad);
                    memcpy(
                        dst + (size_t)y * outputWidth + lpad,
                        src + (size_t)iy * inputWidth + ix + lpad,
                        sizeof(scalar_t) * (outputWidth - rpad - lpad));
                    if (rpad > 0)
                      memset(
                          dst + (size_t)y * outputWidth + outputWidth - rpad,
                          0,
                          sizeof(scalar_t) * rpad);
                  }
                } else {
                  for (x = 0; x < outputWidth; x++) {
                    ix = (int64_t)x * dW - padW + kw;
                    if (ix < 0 || ix >= inputWidth)
                      memset(
                          dst + (size_t)y * outputWidth + x,
                          0,
                          sizeof(scalar_t) * 1);
                    else
                      memcpy(
                          dst + (size_t)y * outputWidth + x,
                          src + (size_t)iy * inputWidth + ix,
                          sizeof(scalar_t) * (1));
                  }
                }
              }
            }
          } else {
            for (y = 0; y < outputHeight; y++) {
              iy = (int64_t)y * dH + kh;
              ix = 0 + kw;
              if (dW == 1)
                memcpy(
                    dst + (size_t)y * outputWidth,
                    src + (size_t)iy * inputWidth + ix,
                    sizeof(scalar_t) * outputWidth);
              else {
                for (x = 0; x < outputWidth; x++)
                  memcpy(
                      dst + (size_t)y * outputWidth + x,
                      src + (size_t)iy * inputWidth + ix + (int64_t)x * dW,
                      sizeof(scalar_t) * (1));
              }
            }
          }
        }
      });
}

void unfolded_copy(
    Tensor finput,
    Tensor input,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int nInputPlane,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight) {
  // This function assumes that
  // kH*kW does not overflow an int
  // nInputPlane*kH*kW does not overflow a int64_t
  // outputHeight*dH does not overflow a int64_t
  // outputWidth*dW does not overflow a int64_t

  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, input.scalar_type(), "unfolded_copy", [&] {
        scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* finput_data = finput.data_ptr<scalar_t>();

        unfolded_copy_kernel(
            input_data,
            finput_data,
            kW,
            kH,
            dW,
            dH,
            padW,
            padH,
            nInputPlane,
            inputWidth,
            inputHeight,
            outputWidth,
            outputHeight);
      });
}

static Tensor view_weight_2d(const Tensor& weight_) {
  Tensor weight = weight_.contiguous();
  if (weight.dim() == 4) {
    int64_t s1 = weight.size(0);
    int64_t s2 = weight.size(1) * weight.size(2) * weight.size(3);
    return weight.view({s1, s2});
  } else {
    return weight;
  }
}

static void conv2d_update_output_frame(
    Tensor input,
    Tensor output,
    Tensor weight,
    Tensor bias,
    Tensor finput,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t nOutputPlane,
    int64_t outputWidth,
    int64_t outputHeight) {
  unfolded_copy(
      finput,
      input,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      nInputPlane,
      inputWidth,
      inputHeight,
      outputWidth,
      outputHeight);

  Tensor output2d = output.view({nOutputPlane, outputHeight * outputWidth});
  if (bias.defined()) {
    for (int64_t i = 0; i < nOutputPlane; i++) {
      output[i].fill_(bias[i].item());
    }
  } else {
    output.zero_();
  }

  output2d.addmm_(weight, finput, 1, 1);
}

} // namespace

std::tuple<Tensor&, Tensor&, Tensor&> conv2d_forward_out_cpu(
    Tensor& output,
    Tensor& finput,
    Tensor& fgrad_input,
    const Tensor& self_,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];

  Tensor weight = view_weight_2d(weight_);

  conv2d_shape_check(
      self_,
      Tensor(),
      weight,
      bias,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      false);

  Tensor self = self_.contiguous();
  int ndim = self.dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  const int64_t nInputPlane = self.size(dimf);
  const int64_t inputHeight = self.size(dimh);
  const int64_t inputWidth = self.size(dimw);
  const int64_t nOutputPlane = weight.size(0);
  const int64_t outputHeight =
      (inputHeight + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t outputWidth =
      (inputWidth + 2 * pad_width - kernel_width) / stride_width + 1;

  if (self.dim() == 3) {
    finput.resize_({kernel_height * kernel_width * nInputPlane,
                    outputHeight * outputWidth});
    output.resize_({nOutputPlane, outputHeight, outputWidth});

    conv2d_update_output_frame(
        self,
        output,
        weight,
        bias,
        finput,
        kernel_width,
        kernel_height,
        stride_width,
        stride_height,
        pad_width,
        pad_height,
        nInputPlane,
        inputWidth,
        inputHeight,
        nOutputPlane,
        outputWidth,
        outputHeight);

  } else {
    int64_t T = self.size(0);

    finput.resize_({T,
                    kernel_width * kernel_height * nInputPlane,
                    outputHeight * outputWidth});
    output.resize_({T, nOutputPlane, outputHeight, outputWidth});

    at::parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
      for (auto t = start; t < end; t++) {
        Tensor input_t = self.select(0, t);
        Tensor output_t = output.select(0, t);
        Tensor finput_t = finput.select(0, t);

        conv2d_update_output_frame(
            input_t,
            output_t,
            weight,
            bias,
            finput_t,
            kernel_width,
            kernel_height,
            stride_width,
            stride_height,
            pad_width,
            pad_height,
            nInputPlane,
            inputWidth,
            inputHeight,
            nOutputPlane,
            outputWidth,
            outputHeight);
      }
    });
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>(output, finput, fgrad_input);
}

std::tuple<Tensor, Tensor, Tensor> conv2d_forward_cpu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto output = at::empty({0}, self.options());
  auto finput = at::empty({0}, self.options());
  auto fgrad_input = at::empty({0}, self.options());
  conv2d_forward_out_cpu(
      output,
      finput,
      fgrad_input,
      self,
      weight,
      kernel_size,
      bias,
      stride,
      padding);
  return std::make_tuple(output, finput, fgrad_input);
}

std::tuple<Tensor&, Tensor&, Tensor&> conv2d_backward_out_cpu(
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const Tensor& finput,
    const Tensor& fgrad_input) {
  if (grad_input.defined()) {
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
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>(
      grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> conv2d_backward_cpu(
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

  conv2d_backward_out_cpu(
      grad_input,
      grad_weight,
      grad_bias,
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      finput,
      fgrad_input);

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

} // namespace native
} // namespace at

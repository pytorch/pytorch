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
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
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

  const int64_t input_height = input.size(dimh);
  const int64_t input_width = input.size(dimw);

  const int64_t exact_input_height = input_height + 2 * padH;
  const int64_t exact_input_width = input_width + 2 * padW;

  TORCH_CHECK(
      exact_input_height >= kH && exact_input_width >= kW,
      "Calculated padded input size per channel: (",
      exact_input_height,
      " x ",
      exact_input_width,
      "). ",
      "Kernel size: (",
      kH,
      " x ",
      kW,
      "). Kernel size can't be greater than actual input size");

  int64_t output_height = div_rtn<int64_t>(exact_input_height - kH, dH) + 1;
  int64_t output_width = div_rtn<int64_t>(exact_input_width - kW, dW) + 1;

  TORCH_CHECK(
      output_width >= 1 && output_height >= 1,
      "Given input size per channel: (",
      input_height,
      " x ",
      input_width,
      "). "
      "Calculated output size per channel: (",
      output_height,
      " x ",
      output_width,
      "). Output size is too small");

  if (weight.defined()) {
    int64_t n_input_plane = weight.size(1);
    if (weight.dim() == 2) {
      n_input_plane /= (kH * kW);
    }
    check_dim_size(input, ndim, dimf, n_input_plane);
  }

  if (grad_output.defined()) {
    if (weight.defined()) {
      int64_t n_output_plane = weight.size(0);
      check_dim_size(grad_output, ndim, dimf, n_output_plane);
    } else if (bias.defined()) {
      TORCH_CHECK(bias.numel() > 0, "non-empty bias tensor expected");
      const int64_t n_output_plane = bias.dim() == 0 ? 1 : bias.size(0);
      check_dim_size(grad_output, ndim, dimf, n_output_plane);
    }
    check_dim_size(grad_output, ndim, dimh, output_height);
    check_dim_size(grad_output, ndim, dimw, output_width);
  }
}

template <typename scalar_t>
void unfolded_acc_kernel(
    scalar_t* finput_data,
    scalar_t* input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  at::parallel_for(0, n_input_plane, 0, [&](int64_t start, int64_t end) {
    for (auto nip = start; nip < end; nip++) {
      int kw, kh, y, x;
      int64_t ix, iy;
      for (kh = 0; kh < kH; kh++) {
        for (kw = 0; kw < kW; kw++) {
          scalar_t* src = finput_data +
              nip * ((size_t)kH * kW * output_height * output_width) +
              kh * ((size_t)kW * output_height * output_width) +
              kw * ((size_t)output_height * output_width);
          scalar_t* dst =
              input_data + nip * ((size_t)input_height * input_width);
          if (padW > 0 || padH > 0) {
            int lpad, rpad;
            for (y = 0; y < output_height; y++) {
              iy = (int64_t)y * dH - padH + kh;
              if (iy < 0 || iy >= input_height) {
              } else {
                if (dW == 1) {
                  ix = 0 - padW + kw;
                  lpad = fmaxf(0, padW - kw);
                  rpad = fmaxf(0, padW - (kW - kw - 1));
                  scalar_t* dst_slice =
                      dst + (size_t)iy * input_width + ix + lpad;
                  THVector_(cadd)(
                      dst_slice,
                      dst_slice,
                      src + (size_t)y * output_width,
                      1,
                      outputWidth);
                  // note: THVector_add could handle 1 value better
                } else {
                  for (x = 0; x < output_width; x++) {
                    ix = (int64_t)x * dW - padW + kw;
                    if (ix < 0 || ix >= input_width) {
                    } else {
                      scalar_t* dst_slice = dst + (size_t)iy * input_width + ix;
                      THVector_(cadd)(
                          dst_slice,
                          dst_slice,
                          src + (size_t)y * output_width + x,
                          1,
                          1);
                    }
                  }
                }
              }
            }
          } else {
            for (y = 0; y < output_height; y++) {
              iy = (int64_t)y * dH + kh;
              ix = 0 + kw;
              if (dW == 1) {
                scalar_t* dst_slice = dst + (size_t)iy * input_width + ix;
                THVector_(cadd)(
                    dst_slice,
                    dst_slice,
                    src + (size_t)y * output_width,
                    1,
                    output_width); // note: THVector_add could handle 1 value
                                   // better
              } else {
                for (x = 0; x < output_width; x++) {
                  scalar_t* dst_slice =
                      dst + (size_t)iy * input_width + ix + x * dW;
                  THVector_(cadd)(
                      dst_slice,
                      dst_slice,
                      src + (size_t)y * output_width + x,
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
}

/* note: due to write issues, this one cannot be parallelized as well as
 * unfolded_copy */
void unfolded_acc(
    Tensor finput,
    Tensor input,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  // This function assumes that
  // output_height*dH does not overflow a int64_t
  // output_width*dW does not overflow a int64_t

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, input.scalar_type(), "unfolded_acc", [&] {
        scalar_t* finput_data = finput.data_ptr<scalar_t>();
        scalar_t* input_data = input.data_ptr<scalar_t>();

        unfolded_acc_kernel(
            finput_data,
            input_data,
            kH,
            kW,
            dH,
            dW,
            padH,
            padW,
            n_input_plane,
            input_height,
            input_width,
            output_height,
            output_width);
      });
}

template <typename scalar_t>
void unfolded_copy_kernel(
    scalar_t* input_data,
    scalar_t* finput_data,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int n_input_plane,
    int input_height,
    int input_width,
    int output_height,
    int output_width) {
  at::parallel_for(
      0, (int64_t)n_input_plane * kH * kW, 0, [&](int64_t start, int64_t end) {
        for (auto k = start; k < end; k++) {
          int64_t nip = k / (kH * kW);
          int64_t rest = k % (kH * kW);
          int64_t kh = rest / kW;
          int64_t kw = rest % kW;
          int x, y;
          int64_t ix, iy;
          scalar_t* dst = finput_data +
              nip * ((size_t)kH * kW * output_height * output_width) +
              kh * ((size_t)kW * output_height * output_width) +
              kw * ((size_t)output_height * output_width);
          scalar_t* src =
              input_data + nip * ((size_t)input_height * input_width);
          if (padW > 0 || padH > 0) {
            int64_t lpad, rpad;
            for (y = 0; y < output_height; y++) {
              iy = (int64_t)y * dH - padH + kh;
              if (iy < 0 || iy >= input_height) {
                memset(
                    dst + (size_t)y * output_width,
                    0,
                    sizeof(scalar_t) * output_width);
              } else {
                if (dW == 1) {
                  ix = 0 - padW + kw;
                  lpad = fmaxf(0, padW - kw);
                  rpad = fmaxf(0, padW - (kW - kw - 1));
                  if (output_width - rpad - lpad <= 0) {
                    memset(
                        dst + (size_t)y * output_width,
                        0,
                        sizeof(scalar_t) * output_width);
                  } else {
                    if (lpad > 0)
                      memset(
                          dst + (size_t)y * output_width,
                          0,
                          sizeof(scalar_t) * lpad);
                    memcpy(
                        dst + (size_t)y * output_width + lpad,
                        src + (size_t)iy * input_width + ix + lpad,
                        sizeof(scalar_t) * (output_width - rpad - lpad));
                    if (rpad > 0)
                      memset(
                          dst + (size_t)y * output_width + output_width - rpad,
                          0,
                          sizeof(scalar_t) * rpad);
                  }
                } else {
                  for (x = 0; x < output_width; x++) {
                    ix = (int64_t)x * dW - padW + kw;
                    if (ix < 0 || ix >= input_width)
                      memset(
                          dst + (size_t)y * output_width + x,
                          0,
                          sizeof(scalar_t) * 1);
                    else
                      memcpy(
                          dst + (size_t)y * output_width + x,
                          src + (size_t)iy * input_width + ix,
                          sizeof(scalar_t) * (1));
                  }
                }
              }
            }
          } else {
            for (y = 0; y < output_height; y++) {
              iy = (int64_t)y * dH + kh;
              ix = 0 + kw;
              if (dW == 1)
                memcpy(
                    dst + (size_t)y * output_width,
                    src + (size_t)iy * input_width + ix,
                    sizeof(scalar_t) * output_width);
              else {
                for (x = 0; x < output_width; x++)
                  memcpy(
                      dst + (size_t)y * output_width + x,
                      src + (size_t)iy * input_width + ix + (int64_t)x * dW,
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
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int n_input_plane,
    int input_height,
    int input_width,
    int output_height,
    int output_width) {
  // This function assumes that
  // kH*kW does not overflow an int
  // n_input_plane*kH*kW does not overflow a int64_t
  // output_height*dH does not overflow a int64_t
  // output_width*dW does not overflow a int64_t

  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, input.scalar_type(), "unfolded_copy", [&] {
        scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* finput_data = finput.data_ptr<scalar_t>();

        unfolded_copy_kernel(
            input_data,
            finput_data,
            kH,
            kW,
            dH,
            dW,
            padH,
            padW,
            n_input_plane,
            input_height,
            input_width,
            output_height,
            output_width);
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
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t n_output_plane,
    int64_t output_height,
    int64_t output_width) {
  unfolded_copy(
      finput,
      input,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      n_input_plane,
      input_height,
      input_width,
      output_height,
      output_width);

  Tensor output2d = output.view({n_output_plane, output_height * output_width});
  if (bias.defined()) {
    for (int64_t i = 0; i < n_output_plane; i++) {
      output[i].fill_(bias[i].item());
    }
  } else {
    output.zero_();
  }

  output2d.addmm_(weight, finput, 1, 1);
}

void conv2d_backward_update_grad_input_frame(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& tweight,
    Tensor& fgrad_input,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width) {
  Tensor grad_output_2d = grad_output.view(
      {grad_output.size(0), grad_output.size(1) * grad_output.size(2)});
  fgrad_input.addmm_(tweight, grad_output_2d, 0, 1);
  grad_input.zero_();

  unfolded_acc(
      fgrad_input,
      grad_input,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      grad_input.size(0),
      grad_input.size(1),
      grad_input.size(2),
      grad_output.size(1),
      grad_output.size(2));
}

void conv2d_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input_,
    const Tensor& weight_,
    Tensor finput,
    Tensor fgrad_input,
    IntArrayRef kernel_size,
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
      input_,
      grad_output_,
      weight,
      Tensor(),
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      false);

  const Tensor input = input_.contiguous();
  const Tensor grad_output = grad_output_.contiguous();

  grad_input.resize_as_(input);
  fgrad_input.resize_as_(finput);

  // depending on the BLAS library, fgrad_input (result tensor) might
  // be left uninitialized on zero alpha, which might lead to weird behavior
  // hence, to be safe, zero it
  fgrad_input.zero_();
  const Tensor tweight = weight.transpose(0, 1);

  if (input.dim() == 3) {
    conv2d_backward_update_grad_input_frame(
        grad_input,
        grad_output,
        tweight,
        fgrad_input,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        pad_height,
        pad_width);
  } else {
    const int64_t T = input.size(0);

    at::parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
      for (auto t = start; t < end; t++) {
        Tensor grad_input_t = grad_input.select(0, t);
        Tensor grad_output_t = grad_output.select(0, t);
        Tensor fgrad_input_t = fgrad_input.select(0, t);

        conv2d_backward_update_grad_input_frame(
            grad_input_t,
            grad_output_t,
            tweight,
            fgrad_input_t,
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            pad_height,
            pad_width);
      }
    });
  }
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
  int64_t ndim = self.dim();
  int64_t dimf = 0;
  int64_t dimh = 1;
  int64_t dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  const int64_t n_input_plane = self.size(dimf);
  const int64_t input_height = self.size(dimh);
  const int64_t input_width = self.size(dimw);
  const int64_t n_output_plane = weight.size(0);
  const int64_t output_height =
      (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =
      (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

  if (self.dim() == 3) {
    // ## AKo: Can we remove this and replace it by a dim() == 4 check/assert??
    // if the call is made through Convolution.cpp this case should
    // not be reachable, since 4D (batch) input in enforced.

    finput.resize_({kernel_height * kernel_width * n_input_plane,
                    output_height * output_width});
    output.resize_({n_output_plane, output_height, output_width});

    conv2d_update_output_frame(
        self,
        output,
        weight,
        bias,
        finput,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        pad_height,
        pad_width,
        n_input_plane,
        input_height,
        input_width,
        n_output_plane,
        output_height,
        output_width);

  } else {
    int64_t T = self.size(0);

    finput.resize_({T,
                    n_input_plane * kernel_height * kernel_width,
                    output_height * output_width});
    output.resize_({T, n_output_plane, output_height, output_width});

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
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            pad_height,
            pad_width,
            n_input_plane,
            input_height,
            input_width,
            n_output_plane,
            output_height,
            output_width);
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
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const Tensor& finput,
    const Tensor& fgrad_input) {
  printf("conv2d_backward_out_cpu\n");

  if (grad_input.defined()) {
    conv2d_backward_out_cpu_template(
        grad_input,
        grad_output,
        input,
        weight,
        finput,
        fgrad_input,
        kernel_size,
        stride,
        padding);
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
    // TODO: Parameter backwards
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
  printf("conv2d_backward_cpu\n");

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

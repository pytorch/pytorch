#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/grad_mode.h>
#include <ATen/div_rtn.h>
#include <ATen/native/Unfold2d.h>

namespace at {
namespace native {

namespace {

static inline void conv2d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& bias,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    bool weight_optional) {
  TORCH_CHECK(
      kernel_width > 0 && kernel_height > 0,
      "kernel size should be greater than zero, but got kernel_height: ",
      kernel_height,
      " kernel_width: ",
      kernel_width);
  TORCH_CHECK(
      stride_width > 0 && stride_height > 0,
      "stride should be greater than zero, but got stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width);

  if (weight.defined()) {
    TORCH_CHECK(
        !weight.numel() == 0 && (weight.dim() == 2 || weight.dim() == 4),
        "non-empty 2D or 4D weight tensor expected, but got: ",
        weight.sizes());
    if (bias.defined()) {
      check_dim_size(bias, 1, 0, weight.size(0));
    }
  } else {
    TORCH_CHECK(weight_optional, "weight tensor is undefined");
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

  const int64_t exact_input_height = input_height + 2 * pad_height;
  const int64_t exact_input_width = input_width + 2 * pad_width;

  TORCH_CHECK(
      exact_input_height >= kernel_height && exact_input_width >= kernel_width,
      "Calculated padded input size per channel: (",
      exact_input_height,
      " x ",
      exact_input_width,
      "). ",
      "Kernel size: (",
      kernel_height,
      " x ",
      kernel_width,
      "). Kernel size can't be greater than actual input size");

  const int64_t output_height =
      div_rtn<int64_t>(exact_input_height - kernel_height, stride_height) + 1;
  const int64_t output_width =
      div_rtn<int64_t>(exact_input_width - kernel_width, stride_width) + 1;

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
      n_input_plane /= (kernel_height * kernel_width);
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

static Tensor view_weight_2d(const Tensor& weight_) {
  Tensor weight = weight_.contiguous();
  if (weight.dim() == 4) {
    const int64_t s1 = weight.size(0);
    const int64_t s2 = weight.size(1) * weight.size(2) * weight.size(3);
    return weight.view({s1, s2});
  } else {
    return weight;
  }
}

static void conv2d_update_output_frame(
    Tensor& input,
    Tensor& output,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& finput,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t n_output_plane,
    int64_t output_height,
    int64_t output_width) {
  unfolded2d_copy_stub(
      kCPU,
      finput,
      input,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      n_input_plane,
      input_height,
      input_width,
      output_height,
      output_width);

  auto output2d =
      output.reshape({n_output_plane, output_height * output_width});
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
    const Tensor& weight,
    Tensor& fgrad_input,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width) {
  auto grad_output_2d = grad_output.reshape(
      {grad_output.size(0), grad_output.size(1) * grad_output.size(2)});
  fgrad_input.addmm_(weight, grad_output_2d, 0, 1);

  grad_input.zero_();
  unfolded2d_acc_stub(
      kCPU,
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
    Tensor& finput,
    Tensor& fgrad_input,
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
    const int64_t batch_size = input.size(0);
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      AutoNonVariableTypeMode guard;
      for (int64_t t = start; t < end; t++) {
        Tensor grad_input_t = grad_input[t];
        Tensor grad_output_t = grad_output[t];
        Tensor fgrad_input_t = fgrad_input[t];
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

void conv2d_backward_parameters_frame(
    Tensor& grad_weight,
    Tensor& grad_bias,
    Tensor& grad_output,
    Tensor& finput) {
  auto grad_output_2d = grad_output.view(
      {grad_output.size(0), grad_output.size(1) * grad_output.size(2)});
  if (grad_weight.defined()) {
    Tensor tfinput = finput.transpose(0, 1);
    grad_weight.addmm_(grad_output_2d, tfinput);
  }

  if (grad_bias.defined()) {
    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        grad_output.scalar_type(),
        "conv2d_backward_parameters",
        [&] {
          auto grad_output_2d_acc = grad_output_2d.accessor<scalar_t, 2>();
          auto grad_bias_acc = grad_bias.accessor<scalar_t, 1>();
          const auto sz = grad_output_2d.size(1);
          for (int64_t i = 0; i < grad_bias.size(0); i++) {
            scalar_t sum = 0;
            for (int64_t k = 0; k < sz; k++) {
              sum += grad_output_2d_acc[i][k];
            }
            grad_bias_acc[i] += sum;
          }
        });
  }
}

static void conv2d_backward_parameters_out_cpu_template(
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& input_,
    const Tensor& grad_output_,
    Tensor finput,
    Tensor fgrad_input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  CheckedFrom c = "conv2d_backward_parameters_cpu";
  auto grad_weight_arg = TensorArg(grad_weight, "grad_weight_arg", 0);
  auto grad_bias_arg = TensorArg(grad_bias, "grad_bias_arg", 0);

  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];

  Tensor grad_weight_2d;
  if (grad_weight.defined()) {
    checkContiguous(c, grad_weight_arg);
    grad_weight_2d = view_weight_2d(grad_weight);
  }

  if (grad_bias.defined()) {
    checkContiguous(c, grad_bias_arg);
  }

  conv2d_shape_check(
      input_,
      grad_output_,
      grad_weight_2d,
      grad_bias,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      true);

  auto input = input_.contiguous();
  auto grad_output = grad_output_.contiguous();

  if (input.dim() == 3) {
    conv2d_backward_parameters_frame(
        grad_weight_2d, grad_bias, grad_output, finput);
  } else {
    const int64_t batch_size = input.size(0);

    for (int64_t t = 0; t < batch_size; t++) {
      Tensor grad_output_t = grad_output[t];
      Tensor finput_t;
      if (grad_weight_2d.defined()) {
        finput_t = finput[t];
      }

      conv2d_backward_parameters_frame(
          grad_weight_2d, grad_bias, grad_output_t, finput_t);
    }
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
  const int64_t ndim = self.dim();
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

  if (ndim == 3) {
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
    const int64_t batch_size = self.size(0);

    finput.resize_({batch_size,
                    n_input_plane * kernel_height * kernel_width,
                    output_height * output_width});
    output.resize_({batch_size, n_output_plane, output_height, output_width});

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      AutoNonVariableTypeMode guard;
      for (int64_t t = start; t < end; t++) {
        Tensor input_t = self[t];
        Tensor output_t = output[t];
        Tensor finput_t = finput[t];
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
  if (grad_input.defined()) {
    conv2d_backward_out_cpu_template(
        grad_input,
        grad_output,
        input,
        weight,
        const_cast<Tensor&>(finput),
        const_cast<Tensor&>(fgrad_input),
        kernel_size,
        stride,
        padding);
  }

  if (grad_weight.defined()) {
    printf("grad_weight\n");
    grad_weight.resize_(weight.sizes());
    grad_weight.zero_();
  }

  if (grad_bias.defined()) {
    grad_bias.resize_({grad_output.size(1)});
    grad_bias.zero_();
  }

  if (grad_weight.defined() || grad_bias.defined()) {
    conv2d_backward_parameters_out_cpu_template(
        grad_weight,
        grad_bias,
        input,
        grad_output,
        finput,
        fgrad_input,
        kernel_size,
        stride,
        padding);
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

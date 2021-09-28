#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/grad_mode.h>
#include <ATen/div_rtn.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/Unfold2d.h>

namespace at {
namespace native {

namespace {

static inline void slow_conv2d_shape_check(
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
        weight.numel() > 0 && (weight.dim() == 2 || weight.dim() == 4),
        "non-empty 2D or 4D weight tensor expected, but got: ",
        weight.sizes());
    if (bias.defined()) {
      check_dim_size(bias, 1, 0, weight.size(0));
    }
  } else {
    TORCH_CHECK(weight_optional, "weight tensor is undefined");
  }

  const int64_t ndim = input.dim();
  const int64_t dim_batch = 0;
  const int64_t dim_planes = 1;
  const int64_t dim_height = 2;
  const int64_t dim_width = 3;

  // Allow for empty batch size but not other dimensions
  bool valid_empty = ndim == 4 && input.size(dim_batch) == 0 &&
      input.size(dim_planes) != 0 && input.size(dim_height) != 0 &&
      input.size(dim_width) != 0;

  TORCH_CHECK(
      (input.numel() > 0 || valid_empty) && ndim == 4,
      "non-empty 4D input tensor expected but got: ",
      input.sizes());

  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);

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
    check_dim_size(input, ndim, dim_planes, n_input_plane);
  }

  if (grad_output.defined()) {
    if (weight.defined()) {
      int64_t n_output_plane = weight.size(0);
      check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
    } else if (bias.defined()) {
      TORCH_CHECK(bias.numel() > 0, "non-empty bias tensor expected");
      const int64_t n_output_plane = bias.dim() == 0 ? 1 : bias.size(0);
      check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
    }
    check_dim_size(grad_output, ndim, dim_height, output_height);
    check_dim_size(grad_output, ndim, dim_width, output_width);
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

template <typename scalar_t>
static void slow_conv2d_update_output_frame(
    TensorAccessor<scalar_t, 3> input,
    TensorAccessor<scalar_t, 3> output,
    TensorAccessor<scalar_t, 2> weight,
    bool has_bias,
    TensorAccessor<scalar_t, 2> finput,
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
  // Note: this is a no_group conv2d
  if ((kernel_height == 1) && (stride_height == 1) && (pad_height == 0) &&
      (kernel_width == 1) && (stride_width == 1) && (pad_width == 0)) {
    // 1x1 kernel, no need to unfold input and finput is already set
  } else {
    unfolded2d_copy_stub(
        kCPU,
        c10::CppTypeToScalarType<scalar_t>::value,
        finput.data(),
        input.data(),
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
  }

  const int beta = has_bias ? 1 : 0;

  // Compute out = weight * input
  // Note gemm expects fortran order, so all 3 matrices are transposed.
  // Swapping argument order cancels this, since C == AB <=> T(C) == T(B)T(A)
  const int64_t m = output_height * output_width;
  const int64_t n = n_output_plane;
  const int64_t k = n_input_plane * kernel_height * kernel_width;

  const int64_t lda = m;
  const int64_t ldb = k;
  const int64_t ldc = m;

  at::native::cpublas::gemm(
      TransposeType::NoTranspose,
      TransposeType::NoTranspose,
      m, n, k,
      static_cast<scalar_t>(1),
      finput.data(), lda,
      weight.data(), ldb,
      static_cast<scalar_t>(beta),
      output.data(), ldc);
}

void slow_conv2d_backward_update_grad_input_frame(
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
  at::mm_out(fgrad_input, weight, grad_output_2d);

  grad_input.zero_();
  TORCH_INTERNAL_ASSERT(fgrad_input.scalar_type() == grad_input.scalar_type());
  unfolded2d_acc_stub(
      kCPU,
      grad_input.scalar_type(),
      fgrad_input.data_ptr(),
      grad_input.data_ptr(),
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

void slow_conv2d_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& finput,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];

  const Tensor weight = view_weight_2d(weight_);
  slow_conv2d_shape_check(
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
  const Tensor tweight = weight.transpose(0, 1);
  const int64_t batch_size = input.size(0);
  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    NoGradGuard no_grad;
    AutoDispatchBelowADInplaceOrView non_variable_type_mode;
    auto fgrad_input = at::empty(finput.sizes().slice(1), finput.options());
    for (int64_t t = start; t < end; t++) {
      Tensor grad_input_t = grad_input[t];
      Tensor grad_output_t = grad_output[t];
      slow_conv2d_backward_update_grad_input_frame(
          grad_input_t,
          grad_output_t,
          tweight,
          fgrad_input,
          kernel_height,
          kernel_width,
          stride_height,
          stride_width,
          pad_height,
          pad_width);
    }
  });
}

void slow_conv2d_backward_weight_frame(
    Tensor& grad_weight,
    Tensor& grad_output,
    const Tensor& finput) {
  auto grad_output_2d = grad_output.view(
      {grad_output.size(0), grad_output.size(1) * grad_output.size(2)});
  const Tensor tfinput = finput.transpose(0, 1);
  grad_weight.addmm_(grad_output_2d, tfinput);
}

static void slow_conv2d_backward_weight_out_cpu_template(
    Tensor& grad_weight,
    const Tensor& input_,
    const Tensor& grad_output_,
    const Tensor& finput,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  CheckedFrom c = "slow_conv2d_backward_parameters_cpu";
  auto grad_weight_arg = TensorArg(grad_weight, "grad_weight_arg", 0);

  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];

  Tensor grad_weight_2d;
  checkContiguous(c, grad_weight_arg);
  grad_weight_2d = view_weight_2d(grad_weight);

  slow_conv2d_shape_check(
      input_,
      grad_output_,
      grad_weight_2d,
      {},
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      true);

  auto input = input_.contiguous();
  auto grad_output = grad_output_.contiguous();

  const int64_t batch_size = input.size(0);
  for (int64_t t = 0; t < batch_size; t++) {
    Tensor grad_output_t = grad_output[t];
    Tensor finput_t;
    if (grad_weight_2d.defined()) {
      finput_t = finput[t];
    }

    slow_conv2d_backward_weight_frame(
        grad_weight_2d, grad_output_t, finput_t);
  }
}

} // namespace

std::tuple<Tensor&, Tensor&> slow_conv2d_forward_out_cpu(
    const Tensor& self,
    const Tensor& weight_,
    IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output,
    Tensor& finput) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];

  const Tensor weight_2d = view_weight_2d(weight_);

  slow_conv2d_shape_check(
      self,
      Tensor(),
      weight_2d,
      bias,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      false);

  const Tensor input = self.contiguous();
  const int64_t dim_planes = 1;
  const int64_t dim_height = 2;
  const int64_t dim_width = 3;

  const int64_t n_input_plane = input.size(dim_planes);
  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);
  const int64_t n_output_plane = weight_2d.size(0);
  const int64_t output_height =
      (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =
      (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

  const int64_t batch_size = input.size(0);

  if ((input.ndimension() == 4) && (kernel_height == 1) && (stride_height == 1) && (pad_height == 0) &&
      (kernel_width == 1) && (stride_width == 1) && (pad_width == 0)) {
    finput =
        input.view({batch_size, n_input_plane, output_height * output_width})
            .detach();
  } else {
     finput.resize_({batch_size,
                  n_input_plane * kernel_height * kernel_width,
                  output_height * output_width});
  }

  output.resize_({batch_size, n_output_plane, output_height, output_width});
  if (bias.defined()) {
    output.copy_(bias.reshape({-1, 1, 1}));
  }
  TORCH_CHECK(output.is_contiguous() && finput.is_contiguous(),
              "slow_conv2d output tensors must be contiguous");

  AT_DISPATCH_ALL_TYPES_AND(kBFloat16, input.scalar_type(), "slow_conv2d_cpu", [&]{
    auto input_a = input.accessor<scalar_t, 4>();
    auto output_a = output.accessor<scalar_t, 4>();
    auto finput_a = finput.accessor<scalar_t, 3>();
    auto weight_2d_a = weight_2d.accessor<scalar_t, 2>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (int64_t t = start; t < end; t++) {
        auto input_t = input_a[t];
        auto output_t = output_a[t];
        auto finput_t = finput_a[t];
        slow_conv2d_update_output_frame(
            input_t,
            output_t,
            weight_2d_a,
            bias.defined(),
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
  });

  return std::tuple<Tensor&, Tensor&>(output, finput);
}

std::tuple<Tensor, Tensor> slow_conv2d_forward_cpu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto output = at::empty({0}, self.options());
  auto finput = at::empty({0}, self.options());
  at::native::slow_conv2d_forward_out_cpu(
      self,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output,
      finput);
  return std::make_tuple(output, finput);
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_backward_out_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const Tensor& finput,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  if (grad_input.defined()) {
    slow_conv2d_backward_out_cpu_template(
        grad_input,
        grad_output,
        self,
        weight,
        finput,
        kernel_size,
        stride,
        padding);
  }

  if (grad_bias.defined()) {
    at::sum_out(grad_bias, grad_output, IntArrayRef{0, 2, 3});
  }

  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes());
    grad_weight.zero_();
    slow_conv2d_backward_weight_out_cpu_template(
        grad_weight,
        self,
        grad_output,
        finput,
        kernel_size,
        stride,
        padding);
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>(
      grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> slow_conv2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const Tensor& finput,
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

  at::native::slow_conv2d_backward_out_cpu(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      finput,
      grad_input,
      grad_weight,
      grad_bias);

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

Tensor & thnn_conv2d_out(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding, Tensor & output) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  Tensor finput = at::empty({0}, self.options());
  return std::get<0>(at::_slow_conv2d_forward_out(output, finput, self, weight, kernel_size, bias, stride, padding));
}

Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  return std::get<0>(at::_slow_conv2d_forward(self, weight, kernel_size, bias, stride, padding));
}

} // namespace native
} // namespace at

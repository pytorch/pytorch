#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/div_rtn.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/Unfold2d.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_slow_conv2d_backward_native.h>
#include <ATen/ops/_slow_conv2d_forward.h>
#include <ATen/ops/_slow_conv2d_forward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/thnn_conv2d_native.h>
#endif

namespace at::native {

namespace {

static Tensor compute_columns2d(
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef kernel_size,
    bool is_channels_last) {
  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];
  const int64_t batch_size = input.size(0);
  const int64_t n_input_plane = input.size(1);
  const int64_t input_height = input.size(2);
  const int64_t input_width = input.size(3);
  const int64_t output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =  (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

  Tensor columns;
  if ((kernel_height == 1) && (stride_height == 1) && (pad_height == 0) &&
      (kernel_width == 1) && (stride_width == 1) && (pad_width == 0)) {
    // Columns are just a view on the input for the 1x1 kernel special case.
    if (is_channels_last) {
      columns = input.as_strided({batch_size, output_height * output_width, n_input_plane},
          {output_height * output_width * n_input_plane, n_input_plane, 1}).detach();
    } else {
      columns = input.view({batch_size, n_input_plane, output_height * output_width}).detach();
    }
  } else {
    int64_t row = is_channels_last ?
        output_height * output_width : n_input_plane * kernel_height * kernel_width;
    int64_t col = is_channels_last ?
        kernel_height * kernel_width * n_input_plane : output_height * output_width;
    columns = at::empty({batch_size, row, col}, input.options());
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "slow_conv2d_cpu", [&]{
      auto input_a = input.accessor<const scalar_t, 4>();
      auto columns_a = columns.accessor<scalar_t, 3>();

      at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
        for (const auto t : c10::irange(start, end)) {
          auto input_t = input_a[t];
          auto columns_t = columns_a[t];
          unfolded2d_copy_stub(
              kCPU,
              c10::CppTypeToScalarType<scalar_t>::value,
              columns_t.data(),
              input_t.data(),
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
              output_width,
              is_channels_last);
        }
      });
    });
  }

  return columns.contiguous();
}

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
  const int64_t dim_planes = 1;
  const int64_t dim_height = 2;
  const int64_t dim_width = 3;

  // Allow for empty batch size and channel size but not other dimensions
  TORCH_CHECK(ndim == 4, "Expected 4D input tensor, but got: ", input.sizes());
  for (const auto dim : c10::irange(2, ndim)) {
    TORCH_CHECK(input.size(dim) != 0,
                "Expected non-zero size for input dimension ", dim,
                ", but got input shape: ", input.sizes(), ". Only the batch and channel dimensions support size 0.");
  }

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
    if (input.size(1) != 0) {
      check_dim_size(input, ndim, dim_planes, n_input_plane);
    }
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

static inline Tensor view_weight_2d(const Tensor& weight_,
    at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) {
  Tensor weight = weight_.contiguous(memory_format);
  if (weight.dim() == 4) {
    const int64_t s1 = weight.size(0);
    const int64_t s2 = weight.size(1) * weight.size(2) * weight.size(3);
    return memory_format == at::MemoryFormat::ChannelsLast
        ? weight.as_strided({s1, s2}, {s2, 1}) // CL: view as {oc, kh*kw*ic}
        : weight.view({s1, s2}); // CF: view as {oc, ic*kh*kw}
  } else {
    return weight;
  }
}

template <typename scalar_t>
static void slow_conv2d_update_output_frame(
    TensorAccessor<const scalar_t, 3> input,
    TensorAccessor<scalar_t, 3> output,
    TensorAccessor<const scalar_t, 2> weight,
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
    int64_t output_width,
    bool is_channels_last) {
  const int beta = has_bias ? 1 : 0;

  // Compute out = weight * input
  // Note gemm expects fortran order, so all 3 matrices are transposed.
  // Swapping argument order cancels this, since C == AB <=> T(C) == T(B)T(A)
  if (is_channels_last) {
    const int64_t m = n_output_plane;
    const int64_t n = output_height * output_width;
    const int64_t k = n_input_plane * kernel_height * kernel_width;

    const int64_t lda = k;
    const int64_t ldb = k;
    const int64_t ldc = m;

    at::native::cpublas::gemm(
        TransposeType::Transpose,
        TransposeType::NoTranspose,
        m, n, k,
        static_cast<scalar_t>(1),
        weight.data(), lda,
        finput.data(), ldb,
        static_cast<scalar_t>(beta),
        output.data(), ldc);
  } else {
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
}

template <typename scalar_t>
void slow_conv2d_backward_update_grad_input_frame(
    TensorAccessor<scalar_t, 3> grad_input,
    TensorAccessor<const scalar_t, 3> grad_output,
    TensorAccessor<const scalar_t, 2> weight,
    scalar_t *fgrad_input,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    bool is_channels_last) {
  // Compute fgrad_input = weight.T * grad_output.reshape({grad_output.shape(0), -1})
  // Note gemm expects fortran order, so all 3 matrices are transposed.
  // Swapping argument order cancels this, since C == AB <=> T(C) == T(B)T(A)
  if (is_channels_last) {
    const int64_t m = weight.size(1);
    const int64_t n = grad_output.size(1) * grad_output.size(2);
    const int64_t k = weight.size(0);

    const int64_t lda = m;
    const int64_t ldb = k;
    const int64_t ldc = m;

    at::native::cpublas::gemm(
        TransposeType::NoTranspose,
        TransposeType::NoTranspose,
        m, n, k,
        static_cast<scalar_t>(1),
        weight.data(), lda,
        grad_output.data(), ldb,
        static_cast<scalar_t>(0),
        fgrad_input, ldc);
  } else {
    const int64_t m = grad_output.size(1) * grad_output.size(2);
    const int64_t n = weight.size(1);
    const int64_t k = weight.size(0);

    const int64_t lda = m;
    const int64_t ldb = n;
    const int64_t ldc = m;

    at::native::cpublas::gemm(
        TransposeType::NoTranspose,
        TransposeType::Transpose,
        m, n, k,
        static_cast<scalar_t>(1),
        grad_output.data(), lda,
        weight.data(), ldb,
        static_cast<scalar_t>(0),
        fgrad_input, ldc);
  }

  unfolded2d_acc_stub(
      kCPU,
      c10::CppTypeToScalarType<scalar_t>::value,
      fgrad_input,
      grad_input.data(),
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
      grad_output.size(2),
      is_channels_last);
}

void slow_conv2d_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input_,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];

  bool use_channels_last = thnn_conv_use_channels_last(input_, weight_);
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  const Tensor weight = view_weight_2d(weight_, memory_format);
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

  const Tensor input = input_.contiguous(memory_format);

  // Compute shape of columnized data excluding batch dim.
  const int64_t batch_size = input.size(0);
  const int64_t n_input_plane = input.size(1);
  const int64_t input_height = input.size(2);
  const int64_t input_width = input.size(3);
  const int64_t output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;
  const int64_t fgrad_input_size = n_input_plane * kernel_height * kernel_width * output_height * output_width;

  const Tensor grad_output = grad_output_.contiguous(memory_format);
  grad_input.resize_as_(input, memory_format);
  grad_input.zero_();
  TORCH_CHECK(grad_input.is_contiguous(memory_format), "slow_conv2d: grad_input must be contiguous");

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, input.scalar_type(), "slow_conv2d_cpu_grad_input", [&] {
    auto grad_output_a = grad_output.accessor<const scalar_t, 4>();
    auto grad_input_a = grad_input.accessor<scalar_t, 4>();
    auto weight_a = weight.accessor<const scalar_t, 2>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      auto fgrad_input = std::make_unique<scalar_t[]>(fgrad_input_size);
      for (const auto t : c10::irange(start, end)) {
        auto grad_input_t = grad_input_a[t];
        auto grad_output_t = grad_output_a[t];
        slow_conv2d_backward_update_grad_input_frame(
            grad_input_t,
            grad_output_t,
            weight_a,
            fgrad_input.get(),
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            pad_height,
            pad_width,
            use_channels_last);
      }
    });
  });
}

template <typename scalar_t>
void slow_conv2d_backward_weight_frame(
    TensorAccessor<scalar_t, 2> grad_weight,
    TensorAccessor<const scalar_t, 3> grad_output,
    TensorAccessor<const scalar_t, 2> finput,
    bool is_channels_last) {
  // Compute grad_weight += grad_output.reshape({grad_output.shape(0), -1}) * finput.T
  // Note gemm expects fortran order, so all 3 matrices are transposed.
  // Swapping argument order cancels this, since C == AB <=> T(C) == T(B)T(A)
  if (is_channels_last) {
    const int64_t m = finput.size(1);
    const int64_t n = grad_output.size(0);
    const int64_t k = grad_output.size(1) * grad_output.size(2);

    const int64_t lda = m;
    const int64_t ldb = n;
    const int64_t ldc = m;

    at::native::cpublas::gemm(
        TransposeType::NoTranspose,
        TransposeType::Transpose,
        m, n, k,
        static_cast<scalar_t>(1),
        finput.data(), lda,
        grad_output.data(), ldb,
        static_cast<scalar_t>(1),
        grad_weight.data(), ldc);
  } else {
    const int64_t m = finput.size(0);
    const int64_t n = grad_output.size(0);
    const int64_t k = grad_output.size(1) * grad_output.size(2);

    const int64_t lda = k;
    const int64_t ldb = k;
    const int64_t ldc = m;

    at::native::cpublas::gemm(
        TransposeType::Transpose,
        TransposeType::NoTranspose,
        m, n, k,
        static_cast<scalar_t>(1),
        finput.data(), lda,
        grad_output.data(), ldb,
        static_cast<scalar_t>(1),
        grad_weight.data(), ldc);
  }
}

static void slow_conv2d_backward_weight_out_cpu_template(
    Tensor& grad_weight,
    const Tensor& input,
    const Tensor& grad_output_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];

  bool use_channels_last = thnn_conv_use_channels_last(input, grad_weight);
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  TORCH_CHECK(grad_weight.is_contiguous(memory_format), "slow_conv2d: grad_weight must be contiguous");
  Tensor grad_weight_2d = view_weight_2d(grad_weight, memory_format);

  slow_conv2d_shape_check(
      input,
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

  auto grad_output = grad_output_.contiguous(memory_format);
  Tensor finput = compute_columns2d(input, padding, stride, kernel_size, use_channels_last);

  const int64_t batch_size = input.size(0);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, input.scalar_type(), "slow_conv2d_cpu_grad_weight", [&] {
    auto grad_output_a = grad_output.accessor<const scalar_t, 4>();
    auto grad_weight_2d_a = grad_weight_2d.accessor<scalar_t, 2>();
    auto finput_a = finput.accessor<const scalar_t, 3>();

    for (const auto t : c10::irange(batch_size)) {
      auto grad_output_t = grad_output_a[t];
      auto finput_t = finput_a[t];

      slow_conv2d_backward_weight_frame(
          grad_weight_2d_a, grad_output_t, finput_t, use_channels_last);
    }
  });
}

} // namespace

Tensor& slow_conv2d_forward_out_cpu(
    const Tensor& self,
    const Tensor& weight_,
    IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
  // See [Note: hacky wrapper removal for optional tensor]

  TORCH_CHECK(kernel_size.size() == 2, "2D kernel_size expected");
  TORCH_CHECK(stride.size() == 2, "2D stride expected");
  TORCH_CHECK(padding.size() == 2, "2D padding expected");

  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];

  bool use_channels_last = thnn_conv_use_channels_last(self, weight_);
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  const Tensor weight_2d = view_weight_2d(weight_, memory_format);

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

  const Tensor input = self.contiguous(memory_format);
  const int64_t batch_size = input.size(0);
  const int64_t n_input_plane = input.size(1);
  const int64_t input_height = input.size(2);
  const int64_t input_width = input.size(3);
  const int64_t n_output_plane = weight_2d.size(0);
  const int64_t output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

  Tensor finput = compute_columns2d(input, padding, stride, kernel_size, use_channels_last);
  output.resize_({batch_size, n_output_plane, output_height, output_width}, memory_format);
  if (bias.defined()) {
    output.copy_(bias.reshape({-1, 1, 1}));
  }
  TORCH_CHECK(output.is_contiguous(memory_format), "slow_conv2d output tensor must be contiguous");

  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "slow_conv2d_cpu", [&]{
    auto input_a = input.accessor<const scalar_t, 4>();
    auto output_a = output.accessor<scalar_t, 4>();
    auto finput_a = finput.accessor<scalar_t, 3>();
    auto weight_2d_a = weight_2d.accessor<const scalar_t, 2>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (const auto t : c10::irange(start, end)) {
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
            output_width,
            use_channels_last);
      }
    });
  });

  return output;
}

Tensor slow_conv2d_forward_cpu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto output = at::empty({0}, self.options());
  at::native::slow_conv2d_forward_out_cpu(
      self,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output);

  return output;
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_backward_out_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  if (grad_input.defined()) {
    slow_conv2d_backward_out_cpu_template(
        grad_input,
        grad_output,
        self,
        weight,
        kernel_size,
        stride,
        padding);
  }

  if (grad_bias.defined()) {
    at::sum_out(grad_bias, grad_output, IntArrayRef{0, 2, 3});
  }

  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes(), weight.suggest_memory_format());
    grad_weight.zero_();
    slow_conv2d_backward_weight_out_cpu_template(
        grad_weight,
        self,
        grad_output,
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
      grad_input,
      grad_weight,
      grad_bias);

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

Tensor & thnn_conv2d_out(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding, Tensor & output) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  return at::_slow_conv2d_forward_out(output, self, weight, kernel_size, bias, stride, padding);
}

Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  return at::_slow_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
}

} // namespace at::native

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/div_rtn.h>
#include <ATen/native/ConvolutionMM3d.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/TransposeType.h>
#include <ATen/native/Unfold3d.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/slow_conv3d_forward.h>
#include <ATen/ops/slow_conv3d_forward_native.h>
#include <ATen/ops/slow_conv3d_native.h>
#include <ATen/ops/sum.h>
#endif

constexpr int64_t CONV3D_GRAIN_SALT = 20;

namespace at {
namespace native {

namespace {

static Tensor compute_columns3d(
    const Tensor& input_,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef kernel_size,
    const int64_t groups) {
  const Tensor input = input_.contiguous();
  const int64_t kernel_depth = kernel_size[0];
  const int64_t kernel_height = kernel_size[1];
  const int64_t kernel_width = kernel_size[2];
  const int64_t pad_depth = padding[0];
  const int64_t pad_height = padding[1];
  const int64_t pad_width = padding[2];
  const int64_t stride_depth = stride[0];
  const int64_t stride_height = stride[1];
  const int64_t stride_width = stride[2];
  const int64_t dim_planes = 1;
  const int64_t dim_depth = 2;
  const int64_t dim_height = 3;
  const int64_t dim_width = 4;
  const int64_t n_input_plane = input.size(dim_planes);
  const int64_t input_depth = input.size(dim_depth);
  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);
  const int64_t output_depth =
      (input_depth + 2 * pad_depth - kernel_depth) / stride_depth + 1;
  const int64_t output_height =
      (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =
      (input_width + 2 * pad_width - kernel_width) / stride_width + 1;
  const int64_t batch_size = input.size(0);

  Tensor columns;
  if ((kernel_depth == 1) && (kernel_height == 1) && (kernel_width == 1) &&
      (pad_depth == 0) && (pad_height == 0) && (pad_width == 0) &&
      (stride_depth == 1) && (stride_height == 1) && (stride_width == 1) && (groups == 1)) {
    // Columns are just a view on the input for this special case.
    columns = input.view({batch_size, n_input_plane, output_height * output_width * output_depth}).detach();
  } else {
    columns = at::empty({batch_size,
                        n_input_plane * kernel_depth * kernel_height * kernel_width,
                        output_depth * output_height * output_width},
                        input.options());

    AT_DISPATCH_ALL_TYPES_AND(kBFloat16, input.scalar_type(), "compute_columns3d", [&] {
      auto input_a = input.accessor<scalar_t, 5>();
      auto columns_a = columns.accessor<scalar_t, 3>();

      at::parallel_for(0, batch_size, CONV3D_GRAIN_SALT, [&](int64_t start, int64_t end) {
        for (const auto t : c10::irange(start, end)) {
          auto input_t = input_a[t];
          auto columns_t = columns_a[t];
          Unfold3dCopyCPU(
            c10::CppTypeToScalarType<scalar_t>::value,
            input_t.data(),
            n_input_plane,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            kernel_depth,
            kernel_height,
            kernel_width,
            stride_depth,
            stride_height,
            stride_width,
            pad_depth,
            pad_height,
            pad_width,
            columns_t.data());
          }
      });
    });
  }

  return columns;
}

static inline void slow_conv3d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& bias,
    int64_t kernel_depth,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_depth,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_depth,
    int64_t pad_height,
    int64_t pad_width,
    int64_t groups,
    bool weight_optional) {
  TORCH_CHECK(
      kernel_width > 0 && kernel_height > 0 && kernel_depth > 0,
      "kernel size should be greater than zero, but got: ",
      kernel_depth,
      " x ",
      kernel_height,
      " x ",
      kernel_width,
      " (TxHxW)");
  TORCH_CHECK(
      stride_width > 0 && stride_height > 0 && stride_depth > 0,
      "stride should be greater than zero, but got: ",
      stride_depth,
      " x ",
      stride_height,
      " x ",
      stride_width,
      " (TxHxW)");
  if (weight.defined()) {
    TORCH_CHECK(
        weight.numel() > 0 && (weight.dim() == 2 || weight.dim() == 5),
        "non-empty 2D or 5D weight tensor expected, but got: ",
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
  const int64_t dim_depth = 2;
  const int64_t dim_height = 3;
  const int64_t dim_width = 4;

  // Allow for empty batch size but not other dimensions
  bool valid_empty = ndim == 5 && input.size(dim_batch) == 0 &&
      input.size(dim_planes) != 0 && input.size(dim_depth) != 0 &&
      input.size(dim_height) != 0 && input.size(dim_width) != 0;

  TORCH_CHECK(
      (input.numel() > 0 || valid_empty) && ndim == 5,
      "non-empty 5D input tensor expected but got: ",
      input.sizes());

  const int64_t input_depth = input.size(dim_depth);
  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);

  const int64_t exact_input_depth = input_depth + 2 * pad_depth;
  const int64_t exact_input_height = input_height + 2 * pad_height;
  const int64_t exact_input_width = input_width + 2 * pad_width;

  TORCH_CHECK(
      exact_input_depth >= kernel_depth &&
          exact_input_height >= kernel_height &&
          exact_input_width >= kernel_width,
      "Calculated padded input size per channel: (",
      exact_input_depth,
      " x ",
      exact_input_height,
      " x ",
      exact_input_width,
      "). ",
      "Kernel size: (",
      kernel_depth,
      " x ",
      kernel_height,
      " x ",
      kernel_width,
      "). Kernel size can't be greater than actual input size");

  const int64_t output_depth =
      div_rtn<int64_t>(exact_input_depth - kernel_depth, stride_depth) + 1;
  const int64_t output_height =
      div_rtn<int64_t>(exact_input_height - kernel_height, stride_height) + 1;
  const int64_t output_width =
      div_rtn<int64_t>(exact_input_width - kernel_width, stride_width) + 1;

  TORCH_CHECK(
      output_depth >= 1 && output_width >= 1 && output_height >= 1,
      "Given input size per channel: (",
      input_depth,
      " x ",
      input_height,
      " x ",
      input_width,
      "). "
      "Calculated output size per channel: (",
      output_depth,
      " x ",
      output_height,
      " x ",
      output_width,
      "). Output size is too small");

  if (weight.defined()) {
    int64_t n_input_plane = weight.size(1);
    if (weight.dim() == 2) {
      n_input_plane /= (kernel_height * kernel_width);
    }
    // to support grouped conv we need to check if input.size(dim_planes)
    // is multiple of weight.size(dim_planes)
    TORCH_CHECK(groups > 0, "none zero group size expected");
    check_dim_size(input, ndim, dim_planes, n_input_plane * groups);
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
    check_dim_size(grad_output, ndim, dim_depth, output_depth);
    check_dim_size(grad_output, ndim, dim_height, output_height);
    check_dim_size(grad_output, ndim, dim_width, output_width);
  }
}

static Tensor view_weight_2d(const Tensor& weight_) {
  Tensor weight = weight_.contiguous();
  if (weight.dim() == 5) {
    const int64_t s1 = weight.size(0);
    const int64_t s2 =
        weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4);
    return weight.view({s1, s2});
  } else {
    return weight;
  }
}

template <typename scalar_t>
static void slow_conv3d_update_output_frame(
    TensorAccessor<scalar_t, 4> input,
    TensorAccessor<scalar_t, 4> output,
    TensorAccessor<scalar_t, 2> weight,
    bool has_bias,
    TensorAccessor<scalar_t, 2> finput,
    int64_t kernel_depth,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_depth,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_depth,
    int64_t pad_height,
    int64_t pad_width,
    int64_t n_input_plane,
    int64_t groups,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t n_output_plane,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  const int beta = has_bias ? 1 : 0;

  // Compute out = weight * input
  // Note gemm expects fortran order, so all 3 matrices are transposed.
  // Swapping argument order cancels this, since C == AB <=> T(C) == T(B)T(A)
  const int64_t m = output_depth * output_height * output_width;
  const int64_t n = (n_output_plane / groups);
  const int64_t k = (n_input_plane / groups) * kernel_depth * kernel_height * kernel_width;

  const int64_t lda = m;
  const int64_t ldb = k;
  const int64_t ldc = m;

  at::native::cpublas::gemm_batched_with_stride(
      TransposeType::NoTranspose,
      TransposeType::NoTranspose,
      groups, m, n, k,
      static_cast<scalar_t>(1),
      finput.data(), lda, finput.stride(0) * k,
      weight.data(), ldb, weight.stride(0) * n,
      static_cast<scalar_t>(beta),
      output.data(), ldc, output.stride(0) * n);
}

template <typename scalar_t>
void slow_conv3d_backward_update_grad_input_frame(
    TensorAccessor<scalar_t, 4> grad_input,
    TensorAccessor<scalar_t, 4> grad_output,
    TensorAccessor<scalar_t, 2> weight,
    TensorAccessor<scalar_t, 2> fgrad_input,
    int64_t kernel_depth,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_depth,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_depth,
    int64_t pad_height,
    int64_t pad_width,
    int64_t groups) {
  // Compute fgrad_input = weight.T * grad_output.reshape({grad_output.shape(0), -1})
  // Note gemm expects fortran order, so all 3 matrices are transposed.
  // Swapping argument order cancels this, since C == AB <=> T(C) == T(B)T(A)
  const int64_t m = grad_output.size(1) * grad_output.size(2) * grad_output.size(3);
  const int64_t n = weight.size(1);
  const int64_t k = weight.size(0) / groups;

  const int64_t lda = m;
  const int64_t ldb = n;
  const int64_t ldc = m;

  at::native::cpublas::gemm_batched_with_stride(
      TransposeType::NoTranspose,
      TransposeType::Transpose,
      groups, m, n, k,
      static_cast<scalar_t>(1),
      grad_output.data(), lda, grad_output.stride(0) * k,
      weight.data(), ldb, weight.stride(0) * k,
      static_cast<scalar_t>(0),
      fgrad_input.data(), ldc, fgrad_input.stride(0) * n);

  Unfold3dAccCPU(
      c10::CppTypeToScalarType<scalar_t>::value,
      fgrad_input.data(),
      grad_input.size(0),
      grad_input.size(1),
      grad_input.size(2),
      grad_input.size(3),
      grad_output.size(1),
      grad_output.size(2),
      grad_output.size(3),
      kernel_depth,
      kernel_height,
      kernel_width,
      stride_depth,
      stride_height,
      stride_width,
      pad_depth,
      pad_height,
      pad_width,
      grad_input.data());
}

void slow_conv3d_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t groups) {
  const int64_t kernel_depth = kernel_size[0];
  const int64_t kernel_height = kernel_size[1];
  const int64_t kernel_width = kernel_size[2];
  const int64_t pad_depth = padding[0];
  const int64_t pad_height = padding[1];
  const int64_t pad_width = padding[2];
  const int64_t stride_depth = stride[0];
  const int64_t stride_height = stride[1];
  const int64_t stride_width = stride[2];

  slow_conv3d_shape_check(
      input,
      grad_output,
      weight,
      Tensor(),
      kernel_depth,
      kernel_height,
      kernel_width,
      stride_depth,
      stride_height,
      stride_width,
      pad_depth,
      pad_height,
      pad_width,
      groups,
      false);

  const Tensor weight2d = view_weight_2d(weight);
  const Tensor grad_output_contiguous = grad_output.contiguous();
  grad_input.resize_as_(input);
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous")

  const int64_t dim_planes = 1;
  const int64_t dim_depth = 2;
  const int64_t dim_height = 3;
  const int64_t dim_width = 4;
  const int64_t n_input_plane = input.size(dim_planes);
  const int64_t input_depth = input.size(dim_depth);
  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);
  const int64_t output_depth =
      (input_depth + 2 * pad_depth - kernel_depth) / stride_depth + 1;
  const int64_t output_height =
      (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =
      (input_width + 2 * pad_width - kernel_width) / stride_width + 1;
  const int64_t batch_size = input.size(0);

  Tensor fgrad_input = at::empty({batch_size,
      n_input_plane * kernel_depth * kernel_height * kernel_width,
      output_depth * output_height * output_width}, input.options());

  AT_DISPATCH_FLOATING_TYPES_AND(
      kBFloat16, input.scalar_type(), "slow_conv3d_cpu_grad_input", [&] {
    at::parallel_for(0, batch_size, CONV3D_GRAIN_SALT,
                    [&](int64_t start, int64_t end) {
        auto grad_input_a = grad_input.accessor<scalar_t, 5>();
        auto grad_output_a = grad_output_contiguous.accessor<scalar_t, 5>();
        auto fgrad_input_a = fgrad_input.accessor<scalar_t, 3>();
        auto weight_2d_a = weight2d.accessor<scalar_t, 2>();

        for (const auto t : c10::irange(start, end)) {
          auto grad_input_t = grad_input_a[t];
          auto grad_output_t = grad_output_a[t];
          auto fgrad_input_t = fgrad_input_a[t];
          slow_conv3d_backward_update_grad_input_frame(
              grad_input_t,
              grad_output_t,
              weight_2d_a,
              fgrad_input_t,
              kernel_depth,
              kernel_height,
              kernel_width,
              stride_depth,
              stride_height,
              stride_width,
              pad_depth,
              pad_height,
              pad_width,
              groups);
        }
    });
  });
}

template <typename scalar_t>
void slow_conv3d_backward_weight_frame(
    TensorAccessor<scalar_t, 2> grad_weight,
    TensorAccessor<scalar_t, 4> grad_output,
    TensorAccessor<scalar_t, 2> finput,
    int64_t groups) {
  // Compute grad_weight += grad_output.reshape({grad_output.shape(0), -1}) * finput.T
  // Note gemm expects fortran order, so all 3 matrices are transposed.
  // Swapping argument order cancels this, since C == AB <=> T(C) == T(B)T(A)
  const int64_t m = grad_weight.size(1);
  const int64_t n = grad_weight.size(0) / groups;
  const int64_t k = grad_output.size(1) * grad_output.size(2) * grad_output.size(3);

  const int64_t lda = k;
  const int64_t ldb = k;
  const int64_t ldc = m;

  at::native::cpublas::gemm_batched_with_stride(
      TransposeType::Transpose,
      TransposeType::NoTranspose,
      groups, m, n, k,
      static_cast<scalar_t>(1),
      finput.data(), lda, finput.stride(0) * m,
      grad_output.data(), ldb, grad_output.stride(0) * n,
      static_cast<scalar_t>(1),
      grad_weight.data(), ldc, grad_weight.stride(0) * n);
}

static void slow_conv3d_backward_parameters_out_cpu_template(
    Tensor& grad_weight,
    const Tensor& input,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t groups) {
  CheckedFrom c = "slow_conv3d_backward_parameters_cpu";
  auto grad_weight_arg = TensorArg(grad_weight, "grad_weight_arg", 0);

  const int64_t kernel_depth = kernel_size[0];
  const int64_t kernel_height = kernel_size[1];
  const int64_t kernel_width = kernel_size[2];
  const int64_t pad_depth = padding[0];
  const int64_t pad_height = padding[1];
  const int64_t pad_width = padding[2];
  const int64_t stride_depth = stride[0];
  const int64_t stride_height = stride[1];
  const int64_t stride_width = stride[2];

  slow_conv3d_shape_check(
      input,
      grad_output,
      grad_weight,
      {},
      kernel_depth,
      kernel_height,
      kernel_width,
      stride_depth,
      stride_height,
      stride_width,
      pad_depth,
      pad_height,
      pad_width,
      groups,
      true);

  Tensor grad_weight_2d = view_weight_2d(grad_weight);
  checkContiguous(c, grad_weight_arg);

  auto grad_output_contiguous = grad_output.contiguous();

  const int64_t batch_size = input.size(0);
  Tensor finput = compute_columns3d(input, stride, padding, kernel_size, groups);

  AT_DISPATCH_FLOATING_TYPES_AND(
      kBFloat16, input.scalar_type(), "slow_conv3d_cpu_grad_weight", [&] {
    auto grad_weight_2d_a = grad_weight_2d.accessor<scalar_t, 2>();
    auto grad_output_a = grad_output_contiguous.accessor<scalar_t, 5>();
    auto finput_a = finput.accessor<scalar_t, 3>();
    for (const auto t : c10::irange(batch_size)) {
      auto grad_output_t = grad_output_a[t];
      auto finput_t = finput_a[t];
      slow_conv3d_backward_weight_frame(
          grad_weight_2d_a, grad_output_t, finput_t, groups);
    }
  });
}

} // namespace

Tensor& slow_conv3d_forward_out_cpu(const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  const int64_t kernel_depth = kernel_size[0];
  const int64_t kernel_height = kernel_size[1];
  const int64_t kernel_width = kernel_size[2];
  const int64_t pad_depth = padding[0];
  const int64_t pad_height = padding[1];
  const int64_t pad_width = padding[2];
  const int64_t stride_depth = stride[0];
  const int64_t stride_height = stride[1];
  const int64_t stride_width = stride[2];

  // TODO: hacky way of deciding the groups
  // Assuming the group size is checked in upstream functions
  const int64_t groups = weight.size(1) > 0 ? self.size(1) / weight.size(1) : 0;

  slow_conv3d_shape_check(
      self,
      Tensor(),
      weight,
      bias,
      kernel_depth,
      kernel_height,
      kernel_width,
      stride_depth,
      stride_height,
      stride_width,
      pad_depth,
      pad_height,
      pad_width,
      groups,
      false);

  const Tensor input = self.contiguous();
  const Tensor weight_2d = view_weight_2d(weight);

  const int64_t dim_planes = 1;
  const int64_t dim_depth = 2;
  const int64_t dim_height = 3;
  const int64_t dim_width = 4;

  const int64_t n_input_plane = input.size(dim_planes);
  const int64_t input_depth = input.size(dim_depth);
  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);
  const int64_t n_output_plane = weight_2d.size(0);
  const int64_t output_depth =
      (input_depth + 2 * pad_depth - kernel_depth) / stride_depth + 1;
  const int64_t output_height =
      (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =
      (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

  Tensor finput = compute_columns3d(input, stride, padding, kernel_size, groups);
  const int64_t batch_size = input.size(0);
  output.resize_(
      {batch_size, n_output_plane, output_depth, output_height, output_width});
  if (bias.defined()) {
    output.copy_(bias.reshape({-1, 1, 1, 1}));
  }

  TORCH_CHECK(output.is_contiguous(), "slow_conv3d output must be contiguous");

  AT_DISPATCH_ALL_TYPES_AND(kBFloat16, input.scalar_type(), "slow_conv3d_cpu", [&] {
    auto input_a = input.accessor<scalar_t, 5>();
    auto output_a = output.accessor<scalar_t, 5>();
    auto finput_a = finput.accessor<scalar_t, 3>();
    auto weight_2d_a = weight_2d.accessor<scalar_t, 2>();

    at::parallel_for(
        0, batch_size, CONV3D_GRAIN_SALT, [&](int64_t start, int64_t end) {
          for (const auto t : c10::irange(start, end)) {
            auto input_t = input_a[t];
            auto output_t = output_a[t];
            auto finput_t = finput_a[t];
            slow_conv3d_update_output_frame(
                input_t,
                output_t,
                weight_2d_a,
                bias.defined(),
                finput_t,
                kernel_depth,
                kernel_height,
                kernel_width,
                stride_depth,
                stride_height,
                stride_width,
                pad_depth,
                pad_height,
                pad_width,
                n_input_plane,
                groups,
                input_depth,
                input_height,
                input_width,
                n_output_plane,
                output_depth,
                output_height,
                output_width);
          }
        });
  });

  return output;
}

Tensor slow_conv3d_forward_cpu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto output = at::empty({0}, self.options());
  at::native::slow_conv3d_forward_out_cpu(
      self,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output);
  return output;
}

static std::tuple<Tensor&, Tensor&, Tensor&> slow_conv3d_backward_out_cpu(const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  // TODO: hacky way of determine the group size
  int64_t groups = self.size(1) / weight.size(1);
  if (grad_input.defined()) {
    slow_conv3d_backward_out_cpu_template(
        grad_input,
        grad_output,
        self,
        weight,
        kernel_size,
        stride,
        padding,
        groups);
  }

  if (grad_bias.defined()) {
    at::sum_out(grad_bias, grad_output, IntArrayRef{0, 2, 3, 4});
  }

  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes());
    grad_weight.zero_();
    slow_conv3d_backward_parameters_out_cpu_template(
        grad_weight,
        self,
        grad_output,
        kernel_size,
        stride,
        padding,
        groups);
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>(
      grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> slow_conv3d_backward_cpu(
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

  at::native::slow_conv3d_backward_out_cpu(
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

Tensor& slow_conv3d_out(const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  return at::slow_conv3d_forward_out(
      output,
      self,
      weight,
      kernel_size,
      bias,
      stride,
      padding);
}

Tensor slow_conv3d(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  return at::slow_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
}

} // namespace native
} // namespace at

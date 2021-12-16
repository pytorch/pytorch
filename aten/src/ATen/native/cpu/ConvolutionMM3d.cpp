#include <ATen/native/ConvolutionMM3d.h>

namespace at {
namespace native {
namespace {

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

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv3d_backward_out_cpu(const Tensor& grad_output,
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

} // namespace

REGISTER_DISPATCH(slow_conv3d_backward_stub, &slow_conv3d_backward_cpu);

} // namespace native
} // namespace at

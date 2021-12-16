#include <ATen/native/ConvolutionMM3d.h>

namespace at {
namespace native {
namespace {

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
  const int64_t groups = self.size(1) / weight.size(1);

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

}} // namespace at::native

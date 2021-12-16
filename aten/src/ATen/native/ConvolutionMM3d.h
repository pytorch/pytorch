#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/grad_mode.h>
#include <ATen/div_rtn.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/Unfold3d.h>
#include <c10/util/irange.h>

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

} // namespace
}} // namespace at::native

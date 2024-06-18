#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>

#include <ATen/core/Tensor.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/im2col.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/slow_conv_transpose2d_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros.h>
#endif

#include <c10/core/TensorOptions.h>
#include <c10/util/irange.h>

namespace at {
namespace {
static inline void slow_conv_transpose2d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& bias,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    int output_padding_height,
    int output_padding_width,
    int dilation_height,
    int dilation_width,
    bool weight_nullable) {
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
  TORCH_CHECK(
      dilation_width > 0 && dilation_height > 0,
      "dilation should be greater than zero, but got dilation_height: ",
      dilation_height,
      ", dilation_width: ",
      dilation_width);
  TORCH_CHECK(
      (output_padding_width < stride_width ||
       output_padding_width < dilation_width) &&
          (output_padding_height < stride_height ||
           output_padding_height < dilation_height),
      "output padding must be smaller than either stride or dilation, but got output_padding_height: ",
      output_padding_height,
      " output_padding_width: ",
      output_padding_width,
      " stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width,
      " dilation_height: ",
      dilation_height,
      " dilation_width: ",
      dilation_width);

  if (weight.defined()) {
    TORCH_CHECK(
        weight.numel() != 0 && (weight.dim() == 2 || weight.dim() == 4),
        "non-empty 2D or 4D weight tensor expected, but got: ",
        weight.sizes());
    if (bias.defined()) {
      check_dim_size(bias, 1, 0, weight.size(1));
    }
  } else if (!weight_nullable) {
    AT_ERROR("weight tensor is expected to be non-nullable");
  }

  int ndim = input.dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  TORCH_CHECK(
      input.numel() != 0 && (ndim == 3 || ndim == 4),
      "non-empty 3D or 4D input tensor expected but got a tensor with size ",
      input.sizes());

  int64_t input_height = input.size(dimh);
  int64_t input_width = input.size(dimw);
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  if (output_width < 1 || output_height < 1) {
    AT_ERROR(
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
  }

  if (weight.defined()) {
    int64_t n_input_plane = weight.size(0);
    check_dim_size(input, ndim, dimf, n_input_plane);
  }

  if (grad_output.defined()) {
    if (weight.defined()) {
      int64_t n_output_plane = weight.size(1);
      check_dim_size(grad_output, ndim, dimf, n_output_plane);
    } else if (bias.defined()) {
      int64_t n_output_plane = bias.size(0);
      check_dim_size(grad_output, ndim, dimf, n_output_plane);
    }
    check_dim_size(grad_output, ndim, dimh, output_height);
    check_dim_size(grad_output, ndim, dimw, output_width);
  }
}
} // namespace

namespace meta {
TORCH_META_FUNC(slow_conv_transpose2d)
(const Tensor& input,
 const Tensor& weight,
 IntArrayRef kernel_size,
 OptionalTensorRef bias_opt,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef output_padding,
 IntArrayRef dilation) {
  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  TORCH_CHECK(
      output_padding.size() == 2,
      "It is expected stride equals to 2, but got size ",
      output_padding.size());

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];
  int64_t output_padding_height = output_padding[0];
  int64_t output_padding_width = output_padding[1];

  slow_conv_transpose2d_shape_check(
      input,
      Tensor(),
      weight,
      bias_opt.getTensorRef(),
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      output_padding_height,
      output_padding_width,
      dilation_height,
      dilation_width,
      false);

  int n_output_plane = weight.size(1);

  bool use_channels_last = native::thnn_conv_use_channels_last(input, weight);
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  Tensor input_ = input.contiguous(memory_format);

  if (input_.dim() == 3) {
    input_.resize_({1, input_.size(0), input_.size(1), input_.size(2)});
  }

  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // Batch size + input planes
  int64_t batch_size = input_.size(0);

  // Resize output
  TensorOptions options(input.options());
  set_output_raw_strided(
      0,
      {batch_size, n_output_plane, output_height, output_width},
      {},
      options.memory_format(memory_format));
}
} // namespace meta

namespace native {

namespace {
void slow_conv_transpose2d_out_cpu_template(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];
  int64_t output_padding_height = output_padding[0];
  int64_t output_padding_width = output_padding[1];

  int n_input_plane = weight.size(0);
  int n_output_plane = weight.size(1);

  bool use_channels_last = thnn_conv_use_channels_last(input, weight);
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  Tensor input_ = input.contiguous(memory_format);
  Tensor weight_ = weight.contiguous(memory_format);
  Tensor bias_ = bias.defined() ? bias.contiguous() : Tensor();

  bool is_batch = false;
  if (input_.dim() == 3) {
    // Force batch
    is_batch = true;
    input_.resize_({1, input.size(0), input.size(1), input.size(2)});
  }

  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // Batch size + input planes
  int64_t batch_size = input_.size(0);

  // Create temporary columns
  Tensor columns = at::empty({0}, input.options());
  if (use_channels_last) {
    columns.resize_({batch_size, input_height * input_width, kernel_height * kernel_width * n_output_plane});
  } else {
    columns.resize_({batch_size, n_output_plane * kernel_height * kernel_width, input_height * input_width});
  }
  columns.zero_();

  // Materialize if COW, since we cannot do so during parallel_for
  output.mutable_data_ptr();

  AT_DISPATCH_FLOATING_TYPES_AND3(at::ScalarType::Long, at::ScalarType::BFloat16,
      at::ScalarType::Half, input.scalar_type(), "slow_conv_transpose2d_out_cpu", [&] {

    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
      // For each elt in batch, do:
      for (const auto elt : c10::irange(begin, end)) {
        // Matrix multiply per output:
        Tensor input_n = input_.select(0, elt);
        Tensor output_n = output.select(0, elt);
        Tensor columns_n = columns.select(0, elt);

        if (use_channels_last) {
          int64_t m = kernel_height * kernel_width * n_output_plane;
          int64_t n = input_height * input_width;
          int64_t k = n_input_plane;

          // column-major matrices
          cpublas::gemm(
              TransposeType::NoTranspose,
              TransposeType::NoTranspose,
              m,
              n,
              k,
              static_cast<scalar_t>(1),
              weight_.const_data_ptr<scalar_t>(),
              m,
              input_n.const_data_ptr<scalar_t>(),
              k,
              static_cast<scalar_t>(0),
              columns_n.mutable_data_ptr<scalar_t>(),
              m);
        } else {
          int64_t m = input_height * input_width;
          int64_t n = n_output_plane * kernel_height * kernel_width;
          int64_t k = n_input_plane;

          // column-major matrices
          cpublas::gemm(
              TransposeType::NoTranspose,
              TransposeType::Transpose,
              m,
              n,
              k,
              static_cast<scalar_t>(1),
              input_n.const_data_ptr<scalar_t>(),
              m,
              weight_.const_data_ptr<scalar_t>(),
              n,
              static_cast<scalar_t>(0),
              columns_n.mutable_data_ptr<scalar_t>(),
              m);
        }

        // Unpack columns back into input:
        col2im<scalar_t>(
            columns_n.const_data_ptr<scalar_t>(),
            n_output_plane,
            output_height,
            output_width,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            pad_height,
            pad_width,
            stride_height,
            stride_width,
            dilation_height,
            dilation_width,
            output_n.data_ptr<scalar_t>(),
            use_channels_last);
      }
    });
  });

  if (bias.defined()) {
    output.add_(bias_.reshape({-1, 1, 1}));
  }

  // Resize output
  if (is_batch) {
    output.resize_({n_output_plane, output_height, output_width});
  }
}

static void slow_conv_transpose2d_backward_out_cpu_template(
    const Tensor& input_,
    const Tensor& grad_output_,
    Tensor& grad_input,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  TORCH_CHECK(
      output_padding.size() == 2,
      "It is expected stride equals to 2, but got size ",
      output_padding.size());

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];
  int64_t output_padding_height = output_padding[0];
  int64_t output_padding_width = output_padding[1];

  int64_t n_input_plane = weight_.size(0);
  int64_t n_output_plane = weight_.size(1);

  bool use_channels_last = thnn_conv_use_channels_last(input_, weight_);
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  slow_conv_transpose2d_shape_check(
      input_,
      grad_output_,
      weight_,
      Tensor(),
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      output_padding_height,
      output_padding_width,
      dilation_height,
      dilation_width,
      false);

  Tensor input = input_.contiguous(memory_format);
  Tensor grad_output = grad_output_.contiguous(memory_format);
  Tensor weight = weight_.contiguous(memory_format);

  bool is_batch = false;
  if (input.dim() == 3) {
    // Force batch
    is_batch = true;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
    grad_output.resize_(
        {1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});
  }

  int64_t input_width = input.size(3);
  int64_t input_height = input.size(2);
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // Batch size + input planes
  int64_t batch_size = input.size(0);

  // Resize output
  grad_input.resize_({batch_size, n_input_plane, input_height, input_width}, memory_format);
  grad_input.zero_();

  // Create temporary columns
  bool need_columns = (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
      stride_width != 1 || pad_height != 0 || pad_width != 0 ||
      dilation_height != 1 || dilation_width != 1);

  Tensor grad_columns = at::empty({0}, input.options());
  if (need_columns) {
    if (use_channels_last) {
      grad_columns.resize_({input_height * input_width, kernel_height * kernel_width * n_output_plane});
    } else {
      grad_columns.resize_({n_output_plane * kernel_height * kernel_width, input_height * input_width});
    }
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half,
      grad_output.scalar_type(), "slow_conv_transpose2d_backward_out_cpu", [&] {
        // Helpers
        Tensor grad_input_n = Tensor();
        Tensor grad_output_n = Tensor();

        // For each elt in batch, do:
        for (const auto elt : c10::irange(batch_size)) {
          // Matrix multiply per sample:
          grad_input_n = grad_input.select(0, elt);
          grad_output_n = grad_output.select(0, elt);

          if (need_columns) {
            // Extract columns:
            im2col<scalar_t>(
                  grad_output_n.const_data_ptr<scalar_t>(),
                  n_output_plane,
                  output_height,
                  output_width,
                  input_height,
                  input_width,
                  kernel_height,
                  kernel_width,
                  pad_height,
                  pad_width,
                  stride_height,
                  stride_width,
                  dilation_height,
                  dilation_width,
                  grad_columns.data_ptr<scalar_t>(),
                  use_channels_last);
          }

          auto gemm_in_ptr = need_columns ? grad_columns.const_data_ptr<scalar_t>()
              : grad_output_n.const_data_ptr<scalar_t>();

          if (use_channels_last) {
            int64_t m = n_input_plane;
            int64_t n = input_height * input_width;
            int64_t k = n_output_plane * kernel_height * kernel_width;

            // column-major matrices
            cpublas::gemm(
                TransposeType::Transpose,
                TransposeType::NoTranspose,
                m,
                n,
                k,
                static_cast<scalar_t>(1),
                weight.const_data_ptr<scalar_t>(),
                k,
                gemm_in_ptr,
                k,
                static_cast<scalar_t>(0),
                grad_input_n.mutable_data_ptr<scalar_t>(),
                m);

          } else {
            int64_t m = input_height * input_width;
            int64_t n = n_input_plane;
            int64_t k = n_output_plane * kernel_height * kernel_width;

            // column-major matrices
            cpublas::gemm(
                TransposeType::NoTranspose,
                TransposeType::NoTranspose,
                m,
                n,
                k,
                static_cast<scalar_t>(1),
                gemm_in_ptr,
                m,
                weight.const_data_ptr<scalar_t>(),
                k,
                static_cast<scalar_t>(0),
                grad_input_n.mutable_data_ptr<scalar_t>(),
                m);
          }
        }

        // Resize output
        if (is_batch) {
          grad_input.resize_({n_input_plane, input_height, input_width});
        }
      });
}

void slow_conv_transpose2d_acc_grad_parameters_cpu(
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& grad_output_,
    Tensor& grad_weight,
    Tensor& grad_bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    int scale_) {
  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  TORCH_CHECK(
      output_padding.size() == 2,
      "It is expected stride equals to 2, but got size ",
      output_padding.size());

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];
  int64_t output_padding_height = output_padding[0];
  int64_t output_padding_width = output_padding[1];

  bool use_channels_last = thnn_conv_use_channels_last(input_, weight_);
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  slow_conv_transpose2d_shape_check(
      input_,
      grad_output_,
      grad_weight,
      grad_bias,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      output_padding_height,
      output_padding_width,
      dilation_height,
      dilation_width,
      true);

  int n_input_plane = weight_.size(0);
  int n_output_plane = weight_.size(1);

  Tensor input = input_.contiguous(memory_format);
  Tensor grad_output = grad_output_.contiguous(memory_format);
  TORCH_CHECK(grad_weight.is_contiguous(memory_format), "grad_weight needs to be contiguous");

  if (input.dim() == 3) {
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
    grad_output.resize_(
        {1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});
  }

  int64_t input_width = input.size(3);
  int64_t input_height = input.size(2);
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // Batch size + input planes
  int64_t batch_size = input.size(0);

  // Resize temporary columns
  bool need_columns = (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
      stride_width != 1 || pad_height != 0 || pad_width != 0 ||
      dilation_height != 1 || dilation_width != 1);

  Tensor columns = at::empty({0}, input.options());
  if (need_columns) {
    if (use_channels_last) {
      columns.resize_({input_height * input_width, kernel_height * kernel_width * n_output_plane});
    } else {
      columns.resize_({n_output_plane * kernel_height * kernel_width, input_height * input_width});
    }
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half,
      input.scalar_type(), "slow_conv_transpose2d_acc_grad_parameters_cpu", [&] {
        // Helpers
        Tensor input_n = Tensor();
        Tensor grad_output_n = Tensor();

        scalar_t scale = static_cast<scalar_t>(scale_);

        // For each elt in batch, do:
        for (const auto elt : c10::irange(batch_size)) {
          // Matrix multiply per output:
          grad_output_n = grad_output.select(0, elt);

          // Do Weight:
          if (grad_weight.defined()) {
            // Matrix multiply per output:
            input_n = input.select(0, elt);

            if (need_columns) {
              // Extract columns:
              im2col<scalar_t>(
                  grad_output_n.const_data_ptr<scalar_t>(),
                  n_output_plane,
                  output_height,
                  output_width,
                  input_height,
                  input_width,
                  kernel_height,
                  kernel_width,
                  pad_height,
                  pad_width,
                  stride_height,
                  stride_width,
                  dilation_height,
                  dilation_width,
                  columns.data_ptr<scalar_t>(),
                  use_channels_last);
            }

            auto gemm_in_ptr = need_columns ? columns.const_data_ptr<scalar_t>()
                : grad_output_n.const_data_ptr<scalar_t>();

            if (use_channels_last) {
              int64_t m = kernel_height * kernel_width * n_output_plane;
              int64_t n = n_input_plane;
              int64_t k = input_height * input_width;

              // column-major matrices
              cpublas::gemm(
                  TransposeType::NoTranspose,
                  TransposeType::Transpose,
                  m,
                  n,
                  k,
                  static_cast<scalar_t>(scale),
                  gemm_in_ptr,
                  m,
                  input_n.const_data_ptr<scalar_t>(),
                  n,
                  static_cast<scalar_t>(1),
                  grad_weight.mutable_data_ptr<scalar_t>(),
                  m);
            } else {
              int64_t m = n_output_plane * kernel_height * kernel_width;
              int64_t n = n_input_plane;
              int64_t k = input_height * input_width;

              // column-major matrices
              cpublas::gemm(
                  TransposeType::Transpose,
                  TransposeType::NoTranspose,
                  m,
                  n,
                  k,
                  static_cast<scalar_t>(scale),
                  gemm_in_ptr,
                  k,
                  input_n.const_data_ptr<scalar_t>(),
                  k,
                  static_cast<scalar_t>(1),
                  grad_weight.mutable_data_ptr<scalar_t>(),
                  m);
            }
          }
        }
      });
}

} // namespace

TORCH_IMPL_FUNC(slow_conv_transpose2d_structured_cpu)
(const Tensor& input,
 const Tensor& weight,
 IntArrayRef kernel_size,
 OptionalTensorRef bias_opt,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef output_padding,
 IntArrayRef dilation,
 const Tensor& output){
  const Tensor& bias = bias_opt.getTensorRef();

  slow_conv_transpose2d_out_cpu_template(
      output,
      input,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output_padding,
      dilation);
 }

static std::tuple<Tensor&, Tensor&, Tensor&> slow_conv_transpose2d_backward_out_cpu(const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  if (grad_input.defined()) {
    slow_conv_transpose2d_backward_out_cpu_template(
        input,
        grad_output,
        grad_input,
        weight,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation);
  }

  if (grad_bias.defined()) {
    at::sum_out(grad_bias, grad_output, IntArrayRef{0, 2, 3});
  }

  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes(), weight.suggest_memory_format());
    grad_weight.zero_();
    slow_conv_transpose2d_acc_grad_parameters_cpu(
        input,
        weight,
        grad_output,
        grad_weight,
        grad_bias,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        1);
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>(
      grad_input, grad_weight, grad_bias);
}

static std::tuple<Tensor, Tensor, Tensor> slow_conv_transpose2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    std::array<bool, 3> output_mask) {
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  } else {
    grad_input = Tensor();
  }

  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  } else {
    grad_weight = Tensor();
  }

  if (output_mask[2]) {
    grad_bias = at::empty({0}, grad_output.options());
  } else {
    grad_bias = Tensor();
  }

  if (grad_input.defined()) {
    slow_conv_transpose2d_backward_out_cpu_template(
        input,
        grad_output,
        grad_input,
        weight,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation);
  }

  if (grad_bias.defined()) {
    at::sum_out(grad_bias, grad_output, IntArrayRef{0, 2, 3});
  }

  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes(), weight.suggest_memory_format());
    grad_weight.zero_();
    slow_conv_transpose2d_acc_grad_parameters_cpu(
        input,
        weight,
        grad_output,
        grad_weight,
        grad_bias,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        1);
  }

  return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}

REGISTER_ALL_CPU_DISPATCH(slow_conv_transpose2d_backward_stub, &slow_conv_transpose2d_backward_cpu);

} // namespace native
} // namespace at

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/im2col.h>
#include <ATen/native/ConvUtils.h>

namespace at {
namespace native {

namespace {

static inline void slow_conv_transpose2d_param_check(
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  TORCH_CHECK(
      kernel_size.size() == 2,
      "Expect kernel_size size of 2, but got ", kernel_size.size());
  TORCH_CHECK(
      dilation.size() == 2,
      "Expect dilation size of 2, but got ", dilation.size());
  TORCH_CHECK(
      padding.size() == 2,
      "Expect padding size of 2, but got ", padding.size());
  TORCH_CHECK(
      stride.size() == 2,
      "Expect stride size of 2, but got ", stride.size());
  TORCH_CHECK(
      output_padding.size() == 2,
      "Expect stride size of 2, but got ", output_padding.size());
}

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

void slow_conv_transposed2d_channels_last(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t dilation_height, int64_t dilation_width,
    Tensor& columns,
    bool skip_col2im) {
  int64_t batch_size = input.size(0);
  int64_t n_input_plane = weight.size(0);
  int64_t n_output_plane = weight.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output.size(2);
  int64_t output_width = output.size(3);

  // resize columns: {N*IH*IW, KH*KW*OC}
  columns.resize_({batch_size * input_height * input_width, kernel_height * kernel_width * n_output_plane});

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
      "slow_conv_transposed2d_channels_last", [&] {
    // Do GEMM in column-major, matrice shape shall be:
    //   input: {IC, N*IH*IW}
    //   weight: {KH*KW*OC, IC}
    //   columns: {KH*KW*OC, N*IH*IW}
    int64_t m = kernel_height * kernel_width * n_output_plane;
    int64_t k = n_input_plane;
    int64_t n = batch_size * input_height * input_width;

    cpublas::gemm(
        cpublas::NoTranspose,
        cpublas::NoTranspose,
        m,
        n,
        k,
        1,
        weight.data_ptr<scalar_t>(),
        m,
        input.data_ptr<scalar_t>(),
        k,
        0,
        skip_col2im ? output.data_ptr<scalar_t>() : columns.data_ptr<scalar_t>(),
        m);
  });

  if (skip_col2im) {
    if (bias.defined()) {
      output.add_(bias.unsqueeze(-1).unsqueeze(-1));
    }
  } else {
    // Init output with bias or zero, before accumulation in col2im
    if (bias.defined()) {
      output.copy_(bias.unsqueeze(-1).unsqueeze(-1));
    } else {
      output.zero_();
    }
    // Unpack columns back into input
    col2im_channels_last_stub(
        kCPU,
        output,
        columns,
        batch_size,
        n_output_plane,
        output_height, output_width,
        input_height, input_width,
        kernel_height, kernel_width,
        pad_height, pad_width,
        stride_height, stride_width,
        dilation_height, dilation_width);
  }
}

void slow_conv_transpose2d_out_cpu_template(
    Tensor& output,
    const Tensor& input_,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    const Tensor& bias_,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& columns_) {
  bool use_channels_last = input_.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      weight_.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  slow_conv_transpose2d_param_check(
      kernel_size,
      stride,
      padding,
      output_padding,
      dilation);

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
      input_,
      Tensor(),
      weight_,
      bias_,
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
  Tensor weight = weight_.contiguous(memory_format);

  Tensor columns = columns_;
  TORCH_CHECK(columns.is_contiguous(), "columns needs to be contiguous");

  bool is_batch = false;
  if (input.dim() == 3) {
    // Force batch
    is_batch = true;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
  }

  auto output_sizes = conv_input_size(
      input.sizes(), weight.sizes(), padding, output_padding, stride, dilation, /*groups*/ 1);

  int64_t batch_size = input.size(0);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output_sizes[2];
  int64_t output_width = output_sizes[3];
  int64_t n_input_plane = weight.size(0);
  int64_t n_output_plane = weight.size(1);

  // Resize output
  output.resize_(output_sizes, memory_format);

  bool skip_col2im = skip_transforming(kernel_size, stride, padding, output_padding);

  if (use_channels_last) {
    slow_conv_transposed2d_channels_last(
      output,
      input,
      weight,
      bias_,
      kernel_height, kernel_width,
      stride_height, stride_width,
      pad_height, pad_width,
      dilation_height, dilation_width,
      columns,
      skip_col2im);
    return;
  }

  // Resize temporary columns
  columns.resize_({batch_size, n_output_plane * kernel_width * kernel_height,
                   input_height * input_width});

  // Parallel for each elt in batch
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    NoGradGuard no_grad;
    AutoNonVariableTypeMode non_variable_type_mode;
    for (int elt = begin; elt < end; elt++) {
      Tensor input_n = input.select(0, elt);
      Tensor output_n = output.select(0, elt);
      Tensor columns_n = columns.select(0, elt);

      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
          "slow_conv_transpose2d_out_cpu", [&] {
        // Do GEMM in column-major, matrice shape shall be:
        //   input: {IC, IH*IW}
        //   weight: {IC, OC*KH*KW}
        //   columns: {OC*KH*KW, IH*IW}
        int64_t m = n_output_plane * kernel_height * kernel_width;
        int64_t n = input_height * input_width;
        int64_t k = n_input_plane;

        cpublas::gemm(
            cpublas::NoTranspose,
            cpublas::Transpose,
            n,
            m,
            k,
            1,
            input_n.data_ptr<scalar_t>(),
            n,
            weight.data_ptr<scalar_t>(),
            m,
            0,
            skip_col2im ? output_n.data_ptr<scalar_t>() : columns_n.data_ptr<scalar_t>(),
            n);
      });

      if (skip_col2im) {
        if (bias_.defined()) {
          output_n.add_(bias_.unsqueeze(-1).unsqueeze(-1));
        }
      } else {
        // Init output with bias or zero, before accumulation in col2im
        if (bias_.defined()) {
          output_n.copy_(bias_.unsqueeze(-1).unsqueeze(-1));
        } else {
          output_n.zero_();
        }
        // Unpack columns back into input
        col2im_stub(
            kCPU,
            output_n,
            columns_n,
            n_output_plane,
            output_height, output_width,
            input_height, input_width,
            kernel_height, kernel_width,
            pad_height, pad_width,
            stride_height, stride_width,
            dilation_height, dilation_width);
      }
    }
  });

  if (is_batch) {
    output.resize_({n_output_plane, output_height, output_width});
  }
}

void slow_conv_transposed2d_backward_channels_last(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& weight,
    int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t dilation_height, int64_t dilation_width,
    Tensor& columns,
    bool skip_im2col) {
  int64_t batch_size = grad_output.size(0);
  int64_t n_input_plane = weight.size(0);
  int64_t n_output_plane = weight.size(1);
  int64_t input_height = grad_input.size(2);
  int64_t input_width = grad_input.size(3);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);

  // resize columns: {N*IH*IW, KH*KW*OC}
  columns.resize_({batch_size * input_height * input_width, kernel_height * kernel_width * n_output_plane});

  if (!skip_im2col) {
    im2col_channels_last_stub(
        kCPU,
        columns,
        grad_output,
        batch_size,
        n_output_plane,
        output_height, output_width,
        input_height, input_width,
        kernel_height, kernel_width,
        pad_height, pad_width,
        stride_height, stride_width,
        dilation_height, dilation_width);
  }

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, grad_output.scalar_type(),
      "slow_conv_transposed2d_backward_channels_last", [&] {
    // Do GEMM in column-major, matrice shape shall be:
    //   weight: {KH*KW*OC, IC}
    //   columns: {KH*KW*OC, N*IH*IW}
    //   grad_input: {IC, N*IH*IW}
    int64_t m = n_input_plane;
    int64_t k = kernel_height * kernel_width * n_output_plane;
    int64_t n = batch_size * input_height * input_width;

    cpublas::gemm(
        cpublas::Transpose,
        cpublas::NoTranspose,
        m,
        n,
        k,
        1,
        weight.data_ptr<scalar_t>(),
        k,
        skip_im2col ? grad_output.data_ptr<scalar_t>() : columns.data_ptr<scalar_t>(),
        k,
        0,
        grad_input.data_ptr<scalar_t>(),
        m);
  });
}

static void slow_conv_transpose2d_backward_out_cpu_template(
    const Tensor& input_,
    const Tensor& grad_output_,
    Tensor& grad_input,
    const Tensor& weight_,
    const Tensor& grad_columns_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  bool use_channels_last = input_.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      weight_.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  slow_conv_transpose2d_param_check(
      kernel_size,
      stride,
      padding,
      output_padding,
      dilation);

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

  Tensor grad_columns = grad_columns_;
  TORCH_CHECK(grad_columns.is_contiguous(), "grad_columns needs to be contiguous");

  bool is_batch = false;
  if (input.dim() == 3) {
    // Force batch
    is_batch = true;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
    grad_output.resize_({1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});
  }

  int64_t batch_size = input.size(0);
  int64_t input_width = input.size(3);
  int64_t input_height = input.size(2);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);
  int64_t n_input_plane = weight.size(0);
  int64_t n_output_plane = weight.size(1);

  // Resize output
  grad_input.resize_({batch_size, n_input_plane, input_height, input_width}, memory_format);
  grad_input.zero_();

  bool skip_im2col = skip_transforming(kernel_size, stride, padding, output_padding);

  if (use_channels_last) {
    slow_conv_transposed2d_backward_channels_last(
        grad_input,
        grad_output,
        weight,
        kernel_height, kernel_width,
        stride_height, stride_width,
        pad_height, pad_width,
        dilation_height, dilation_width,
        grad_columns,
        skip_im2col);
    return;
  }

  // Resize temporary columns
  grad_columns.resize_({batch_size, n_output_plane * kernel_width * kernel_height,
                        input_height * input_width});

  // Parallel for each elt in batch
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    NoGradGuard no_grad;
    AutoNonVariableTypeMode non_variable_type_mode;
    for (int64_t elt = begin; elt < end; elt++) {
      Tensor grad_input_n = grad_input.select(0, elt);
      Tensor grad_output_n = grad_output.select(0, elt);
      Tensor grad_columns_n = grad_columns.select(0, elt);

      if (!skip_im2col) {
        // Extract columns:
        im2col_stub(kCPU,
            grad_columns_n,
            grad_output_n,
            n_output_plane,
            output_height, output_width,
            input_height, input_width,
            kernel_height, kernel_width,
            pad_height, pad_width,
            stride_height, stride_width,
            dilation_height, dilation_width);
      }

      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
          "slow_conv_transpose2d_backward_out_cpu", [&] {
        // Do GEMM in column-major, matrice shape shall be:
        //   columns: {OC*KH*KW,IH*IW}
        //   weight: {IC, OC*KH*KW}
        //   grad_input: {IC, IH*IW}
        int64_t m = n_input_plane;
        int64_t n = input_height * input_width;
        int64_t k = n_output_plane * kernel_height * kernel_width;

        cpublas::gemm(
            cpublas::NoTranspose,
            cpublas::NoTranspose,
            n,
            m,
            k,
            1,
            skip_im2col ? grad_output_n.data_ptr<scalar_t>() : grad_columns_n.data_ptr<scalar_t>(),
            n,
            weight.data_ptr<scalar_t>(),
            k,
            0,
            grad_input_n.data_ptr<scalar_t>(),
            n);
      });
    }
  });

  if (is_batch) {
    grad_input.resize_({n_input_plane, input_height, input_width});
  }
}

void slow_conv_transposed_acc_grad_channels_last(
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& input,
    const Tensor& grad_output,
    int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t dilation_height, int64_t dilation_width,
    Tensor& columns) {
  int64_t batch_size = input.size(0);
  int64_t n_input_plane = grad_weight.size(0);
  int64_t n_output_plane = grad_weight.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);

  if (grad_weight.defined()) {
    // resize columns: {N*IH*IW, KH*KW*OC}
    columns.resize_({batch_size * input_height * input_width, kernel_height * kernel_width * n_output_plane});

    im2col_channels_last_stub(
        kCPU,
        columns,
        grad_output,
        batch_size,
        n_output_plane,
        output_height, output_width,
        input_height, input_width,
        kernel_height, kernel_width,
        pad_height, pad_width,
        stride_height, stride_width,
        dilation_height, dilation_width);

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
        "slow_conv_transposed_acc_grad_channels_last", [&] {
      // Do GEMM in column-major, matrice shape shall be:
      //   columns: {KH*KW*OC, N*IH*IW}
      //   input: {IC, N*IH*IW}
      //   grad_weight: {KH*KW*OC, IC}
      int64_t m = kernel_height * kernel_width * n_output_plane;
      int64_t k = batch_size * input_height * input_width;
      int64_t n = n_input_plane;

      cpublas::gemm(
          cpublas::NoTranspose,
          cpublas::Transpose,
          m,
          n,
          k,
          1,
          columns.data_ptr<scalar_t>(),
          m,
          input.data_ptr<scalar_t>(),
          n,
          1,
          grad_weight.data_ptr<scalar_t>(),
          m);
    });
  }

  if (grad_bias.defined()) {
    // 2d view of grad_output
    const int64_t s1 = batch_size *output_height * output_width;
    const int64_t s2 = n_output_plane;
    auto grad_output_2d = grad_output.as_strided({s1, s2}, {s2, 1});
    grad_bias.copy_(grad_output_2d.sum(0));
  }
}

void slow_conv_transpose2d_acc_grad_parameters_cpu(
    const Tensor& input_,
    const Tensor& grad_output_,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& columns_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  bool use_channels_last = input_.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  slow_conv_transpose2d_param_check(
      kernel_size,
      stride,
      padding,
      output_padding,
      dilation);

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

  if (!grad_weight.defined() && !grad_bias.defined() ) {
    return;
  }

  Tensor input = input_.contiguous(memory_format);
  Tensor grad_output = grad_output_.contiguous(memory_format);

  Tensor columns = columns_;
  TORCH_CHECK(columns.is_contiguous(), "columns needs to be contiguous");
  if (grad_weight.defined()) {
    TORCH_CHECK(grad_weight.is_contiguous(memory_format), "grad_weight needs to be contiguous");
  }
  if (grad_bias.defined()) {
    TORCH_CHECK(grad_bias.is_contiguous(), "grad_bias needs to be contiguous");
  }

  bool is_batch = false;
  if (input.dim() == 3) {
    // Force batch
    is_batch = true;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
    grad_output.resize_({1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});
  }

  int64_t batch_size = input.size(0);
  int64_t input_width = input.size(3);
  int64_t input_height = input.size(2);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);
  int64_t n_input_plane = input.size(1);
  int64_t n_output_plane = grad_output.size(1);

  bool skip_im2col = skip_transforming(kernel_size, stride, padding, output_padding);

  if (use_channels_last) {
    slow_conv_transposed_acc_grad_channels_last(
        grad_weight,
        grad_bias,
        input,
        grad_output,
        kernel_height, kernel_width,
        stride_height, stride_width,
        pad_height, pad_width,
        dilation_height, dilation_width,
        columns);
    return;
  }

  // Resize temporary columns
  columns.resize_({n_output_plane * kernel_width * kernel_height,
                   input_height * input_width});

  // For each elt in batch, sequentially do:
  for (int elt = 0; elt < batch_size; elt++) {
    Tensor grad_output_n = grad_output.select(0, elt);

    if (grad_weight.defined()) {
      Tensor input_n = input.select(0, elt);

      if (!skip_im2col) {
        im2col_stub(
            kCPU,
            columns,
            grad_output_n,
            n_output_plane,
            output_height, output_width,
            input_height, input_width,
            kernel_height, kernel_width,
            pad_height, pad_width,
            stride_height, stride_width,
            dilation_height, dilation_width);
      }

      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
          "slow_conv_transpose2d_acc_grad_parameters_cpu", [&] {
        // Do GEMM in column-major, matrice shape shall be:
        //   columns: {OC*KH*KW, IH*IW}
        //   input: {IC, IH*IW}
        //   grad_weight: {IC, OC*KH*KW}
        int64_t n = n_output_plane * kernel_height * kernel_width;
        int64_t m = n_input_plane;
        int64_t k = input_height * input_width;

        cpublas::gemm(
            cpublas::Transpose,
            cpublas::NoTranspose,
            n,
            m,
            k,
            1,
            skip_im2col ? grad_output_n.data_ptr<scalar_t>() : columns.data_ptr<scalar_t>(),
            k,
            input_n.data_ptr<scalar_t>(),
            k,
            1,
            grad_weight.data_ptr<scalar_t>(),
            n);
      });
    }

    if (grad_bias.defined()) {
      Tensor grad_output_2d = grad_output_n.view({n_output_plane, -1});
      grad_bias.add_(grad_output_2d.sum(1));
    }
  }
}

} // namespace

Tensor& slow_conv_transpose2d_out_cpu(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  Tensor columns = at::empty({0}, input.options());

  slow_conv_transpose2d_out_cpu_template(
      output,
      input,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      columns);

  return output;
}

Tensor slow_conv_transpose2d_cpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  Tensor output = at::empty({0}, input.options());
  Tensor columns = at::empty({0}, input.options());

  slow_conv_transpose2d_out_cpu_template(
      output,
      input,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      columns);

  return output;
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv_transpose2d_backward_out_cpu(
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    const Tensor& columns) {
  if (grad_input.defined()) {
    slow_conv_transpose2d_backward_out_cpu_template(
        input,
        grad_output,
        grad_input,
        weight,
        columns,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation);
  }

  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes(), input.suggest_memory_format());
    grad_weight.zero_();
  }

  if (grad_bias.defined()) {
    grad_bias.resize_({weight.size(1)});
    grad_bias.zero_();
  }

  if (grad_weight.defined() || grad_bias.defined()) {
    slow_conv_transpose2d_acc_grad_parameters_cpu(
        input,
        grad_output,
        grad_weight,
        grad_bias,
        columns,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation);
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>(
      grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> slow_conv_transpose2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    const Tensor& columns,
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

  if (grad_input.defined()) {
    slow_conv_transpose2d_backward_out_cpu_template(
        input,
        grad_output,
        grad_input,
        weight,
        columns,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation);
  }

  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes(), input.suggest_memory_format());
    grad_weight.zero_();
  }

  if (grad_bias.defined()) {
    grad_bias.resize_({weight.size(1)});
    grad_bias.zero_();
  }

  if (grad_weight.defined() || grad_bias.defined()) {
    slow_conv_transpose2d_acc_grad_parameters_cpu(
        input,
        grad_output,
        grad_weight,
        grad_bias,
        columns,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation);
  }

  return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}

DEFINE_DISPATCH(col2im_stub);
DEFINE_DISPATCH(im2col_stub);
DEFINE_DISPATCH(col2im_channels_last_stub);
DEFINE_DISPATCH(im2col_channels_last_stub);

} // namespace native
} // namespace at

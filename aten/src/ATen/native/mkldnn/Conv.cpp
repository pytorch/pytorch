#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups) {
  AT_ERROR("mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, std::array<bool, 3> output_mask) {
  AT_ERROR("mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_convolution_transpose(
    const Tensor& input, const Tensor& weight_t, const Tensor& bias,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups) {
  AT_ERROR("mkldnn_convolution_transpose: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_convolution_transpose_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_transpose_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_convolution_transpose_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_transpose_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_transpose_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, std::array<bool, 3> output_mask) {
  AT_ERROR("mkldnn_convolution_transpose_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/mkldnn/Runtime.h>

namespace at { namespace native {

namespace {

constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int input_channels_dim = 1;
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;

static std::vector<int64_t> conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {

  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

static std::vector<int64_t> conv_input_size(
    IntArrayRef output_size, IntArrayRef weight_size, IntArrayRef padding,
    IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {

  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_output_channels_dim] * groups;

  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2])
                        + kernel + output_padding[d - 2];
  }
  return input_size;
}

struct ConvolutionParams {
  int64_t dim;
  int64_t input_size[2 + max_dim];
  int64_t weight_size[2 + max_dim];
  int64_t output_size[2 + max_dim];
  int64_t padding[max_dim];
  int64_t stride[max_dim];
  int64_t dilation[max_dim];
  int64_t groups;
  bool has_bias;
  bool transpose;
};

void setConvolutionParams(
    ConvolutionParams* params, const Tensor& input, const Tensor& weight,
    const Tensor& output, IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, bool has_bias, bool transpose) {

  memset(params, 0, sizeof(ConvolutionParams));

  params->dim = input.dim();
  for (int64_t i = 0; i < params->dim; ++i) {
    params->input_size[i] = input.size(i);
    params->weight_size[i] = weight.size(i);
    params->output_size[i] = output.size(i);
  }
  for (size_t i = 0; i < padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  params->groups = groups;
  params->has_bias = has_bias;
  params->transpose = transpose;
}

struct ConvolutionArgs {
  ConvolutionParams params;
  tensor::dims input_tz;
  tensor::dims weight_tz;
  tensor::dims bias_tz;
  tensor::dims output_tz;
  tensor::dims _stride;
  tensor::dims _dilation;
  tensor::dims _padding;
  tensor::dims _padding_r;

  ConvolutionArgs(const Tensor& input, const Tensor& weight, const Tensor& output,
      IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
      int64_t groups, bool has_bias, bool transpose) {

    setConvolutionParams(&params, input, weight, output,
        padding, stride, dilation, groups, has_bias, transpose);

    if (groups != 1) weight_tz.push_back(groups);
    for (int64_t i = 0; i < input.dim(); ++i) {
      input_tz.push_back(params.input_size[i]);
      weight_tz.push_back(params.weight_size[i]);
      output_tz.push_back(params.output_size[i]);
    }
    if (groups != 1) {
      if (transpose) {
        weight_tz[weight_input_channels_dim + 1] /= groups;
      } else {
        weight_tz[weight_output_channels_dim + 1] /= groups;
      }
    }
    bias_tz.push_back(output.size(output_channels_dim));
    for (size_t k = 0; k < padding.size(); ++k) {
      _stride.push_back(stride[k]);
      _dilation.push_back(dilation[k]);
      _padding.push_back(padding[k]);
      if (transpose) {
        _padding_r.push_back((input.size(k + 2) - 1) * stride[k] - output.size(k + 2)
          + ((weight.size(k + 2) - 1) * dilation[k] + 1) + output_padding[k] - padding[k]);
      } else {
        _padding_r.push_back((output.size(k + 2) - 1) * stride[k] - input.size(k + 2) +
          ((weight.size(k + 2) - 1) * dilation[k] + 1) - padding[k]);
      }
    }
  }
};

} // anonymous namespace

Tensor mkldnn_convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups) {

  auto output = at::empty(conv_output_size(
    input.sizes(), weight.sizes(), padding, stride, dilation, groups), input.options());

  ConvolutionArgs args(input, weight, output, padding, padding,
      stride, dilation, groups, bias.defined(), false);

  auto dtype = get_mkldnn_dtype(input);
  desc x_desc(args.input_tz, dtype);
  desc weight_desc(args.weight_tz, dtype);

  itensor x_, weight_, bias_, y_, y/*user layout*/;

  x_.init(x_desc, input.data_ptr());
  weight_.init(weight_desc, weight.data_ptr());

  desc y_desc(args.output_tz, x_.get_data_type());
  y.init(y_desc, output.data_ptr());

  if (args.params.has_bias) {
    desc bias_desc(args.bias_tz, dtype);
    bias_.init(bias_desc, bias.data_ptr());
    convolution_forward::compute(x_, weight_, bias_, args.output_tz, y_,
        args._stride, args._dilation, args._padding, args._padding_r);
  } else {
    convolution_forward::compute(x_, weight_, args.output_tz, y_,
        args._stride, args._dilation, args._padding, args._padding_r);
  }
  reorder::compute(y_, y);

  return output;
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool bias_defined) {

  auto grad_input = at::empty(input_size, grad_output.options());

  ConvolutionArgs args(grad_input, weight, grad_output, padding, padding,
      stride, dilation, groups, bias_defined, false);

  auto dtype = get_mkldnn_dtype(grad_output);
  desc grady_desc(args.output_tz, dtype);
  desc weight_desc(args.weight_tz, dtype);

  itensor grady_, weight_, gradx_, gradx/*user layout*/;

  grady_.init(grady_desc, grad_output.data_ptr());
  weight_.init(weight_desc, weight.data_ptr());

  desc gradx_desc(args.input_tz, grady_.get_data_type());
  gradx.init(gradx_desc, grad_input.data_ptr());

  convolution_backward_data::compute(grady_, weight_, args.input_tz, gradx_,
      args._stride, args._dilation, args._padding, args._padding_r);

  reorder::compute(gradx_, gradx);

  return grad_input;
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool bias_defined) {

  auto grad_weight = at::empty(weight_size, grad_output.options());
  Tensor grad_bias;
  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
  }

  ConvolutionArgs args(input, grad_weight, grad_output, padding,
      padding, stride, dilation, groups, bias_defined, false);

  auto dtype = get_mkldnn_dtype(input);
  desc x_desc(args.input_tz, dtype);
  desc grady_desc(args.output_tz, dtype);

  itensor x_, grady_, gradw_, gradw/*user layout*/, gradb_, gradb/*user layout*/;

  x_.init(x_desc, input.data_ptr());
  grady_.init(grady_desc, grad_output.data_ptr());

  desc gradw_desc(args.weight_tz, grady_.get_data_type());
  gradw.init(gradw_desc, grad_weight.data_ptr());

  if (bias_defined) {
    desc gradb_desc(args.bias_tz, grady_.get_data_type());
    gradb.init(gradb_desc, grad_bias.data_ptr());
    convolution_backward_weights::compute(x_, grady_, args.weight_tz, gradw_, gradb_,
        args._stride, args._dilation, args._padding, args._padding_r);
    reorder::compute(gradb_, gradb);
  } else {
    convolution_backward_weights::compute(x_, grady_, args.weight_tz, gradw_,
        args._stride, args._dilation, args._padding, args._padding_r);
  }
  reorder::compute(gradw_, gradw);

  return std::tuple<at::Tensor, at::Tensor>{grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_convolution_backward_input(
        input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_convolution_backward_weights(
        weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

Tensor mkldnn_convolution_transpose(
    const Tensor& input, const Tensor& weight_t, const Tensor& bias,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups) {

  //MKLDNN doesn't support iohw or giohw weight format for now
  auto weight = weight_t.transpose(0,1).contiguous();
  auto output = at::empty(conv_input_size(input.sizes(), weight.sizes(),
    padding, output_padding, stride, dilation, groups), input.options());

  ConvolutionArgs args(input, weight, output, padding, output_padding,
      stride, dilation, groups, bias.defined(), true);

  auto dtype = get_mkldnn_dtype(input);
  desc x_desc(args.input_tz, dtype);
  desc weight_desc(args.weight_tz, dtype);

  itensor x_, weight_, bias_, y_, y/*user layout*/;

  x_.init(x_desc, input.data_ptr());
  weight_.init(weight_desc, weight.data_ptr());

  desc y_desc(args.output_tz, x_.get_data_type());
  y.init(y_desc, output.data_ptr());

  if (args.params.has_bias) {
    desc bias_desc(args.bias_tz, dtype);
    bias_.init(bias_desc, bias.data_ptr());
    convolution_transpose_forward::compute(x_, weight_, bias_, args.output_tz, y_,
        args._stride, args._padding, args._padding_r);
  } else {
    convolution_transpose_forward::compute(x_, weight_, args.output_tz, y_,
        args._stride, args._padding, args._padding_r);
  }
  reorder::compute(y_, y);

  return output;
}

Tensor mkldnn_convolution_transpose_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, bool bias_defined) {

  auto grad_input = at::empty(input_size, grad_output.options());
  ConvolutionArgs args(grad_input, weight, grad_output, padding, output_padding,
      stride, dilation, groups, bias_defined, true);

  auto dtype = get_mkldnn_dtype(grad_output);
  desc grady_desc(args.output_tz, dtype);
  desc weight_desc(args.weight_tz, dtype);

  itensor grady_, weight_, gradx_, gradx/*user layout*/;

  grady_.init(grady_desc, grad_output.data_ptr());
  weight_.init(weight_desc, weight.data_ptr());

  desc gradx_desc(args.input_tz, grady_.get_data_type());
  gradx.init(gradx_desc, grad_input.data_ptr());

  convolution_transpose_backward_data::compute(grady_, weight_, args.input_tz,
      gradx_, args._stride, args._padding, args._padding_r);

  reorder::compute(gradx_, gradx);

  return grad_input;
}

std::tuple<Tensor, Tensor> mkldnn_convolution_transpose_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, bool bias_defined) {

  auto grad_weight = at::empty(weight_size, grad_output.options());
  Tensor grad_bias;
  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
  }

  ConvolutionArgs args(input, grad_weight, grad_output, padding,
      output_padding, stride, dilation, groups, bias_defined, true);

  auto dtype = get_mkldnn_dtype(input);
  desc x_desc(args.input_tz, dtype);
  desc grady_desc(args.output_tz, dtype);

  itensor x_, grady_, gradw_, gradw/*user layout*/, gradb_, gradb/*user layout*/;
  x_.init(x_desc, input.data_ptr());
  grady_.init(grady_desc, grad_output.data_ptr());

  desc gradw_desc(args.weight_tz, grady_.get_data_type());
  gradw.init(gradw_desc, grad_weight.data_ptr());

  if (bias_defined) {
    desc gradb_desc(args.bias_tz, grady_.get_data_type());
    gradb.init(gradb_desc, grad_bias.data_ptr());
    convolution_transpose_backward_weights::compute(x_, grady_, args.weight_tz, gradw_, gradb_,
        args._stride, args._padding, args._padding_r);
    reorder::compute(gradb_, gradb);
  } else {
    convolution_transpose_backward_weights::compute(x_, grady_, args.weight_tz, gradw_,
        args._stride, args._padding, args._padding_r);
  }

  reorder::compute(gradw_, gradw);
  auto grad_weight_t = grad_weight.transpose(0 , 1);

  return std::tuple<Tensor, Tensor>{grad_weight_t, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_transpose_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();
  auto weight = weight_t.transpose(0,1).contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_convolution_transpose_backward_input(input.sizes(), grad_output,
        weight, padding, output_padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_convolution_transpose_backward_weights(weight.sizes(),
        grad_output, input, padding, output_padding, stride, dilation, groups, output_mask[2]);
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

}}  // namespace at::native

#endif

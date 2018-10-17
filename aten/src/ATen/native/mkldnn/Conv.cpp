#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

at::Tensor mkldnn_convolution(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    IntList padding, IntList stride, IntList dilation, int64_t groups) {
  AT_ERROR("mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

at::Tensor mkldnn_convolution_backward_input(
    IntList input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation, int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<at::Tensor,at::Tensor> mkldnn_convolution_backward_weights(
    IntList weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntList padding, IntList stride, IntList dilation, int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation, int64_t groups, std::array<bool,3> output_mask) {
  AT_ERROR("mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/mkldnn/Runtime.h>

using namespace mkldnn;

namespace at { namespace native {

constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int input_channels_dim = 1;
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;

static std::vector<int64_t> conv_output_size(
    IntList input_size, IntList weight_size,
    IntList padding, IntList stride, IntList dilation, int64_t groups)
{
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

at::Tensor mkldnn_convolution(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    IntList padding, IntList stride, IntList dilation, int64_t groups)
{
  auto output = at::empty(conv_output_size(
    input.sizes(), weight.sizes(), padding, stride, dilation, groups), input.options());

  auto cpu_engine = CpuEngine::Instance().get_engine();
  // TODO: fn_train flag?
  auto conv_prop = prop_kind::forward;
  // TODO: winograd?
  auto conv_algo = algorithm::convolution_direct;

  auto dim = input.dim();
  auto kdim = dim - 2;
  auto depth_dim = 2;
  auto height_dim = (dim == 5) ? 3 : 2;
  auto weight_dim = (dim == 5) ? 4 : 3;

  int32_t g = groups;

  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t id = input.size(depth_dim);
  int32_t ih = input.size(height_dim);
  int32_t iw = input.size(weight_dim);

  int32_t oc = output.size(1);
  int32_t od = output.size(depth_dim);
  int32_t oh = output.size(height_dim);
  int32_t ow = output.size(weight_dim);

  int32_t kd = weight.size(depth_dim);
  int32_t kh = weight.size(height_dim);
  int32_t kw = weight.size(weight_dim);

  bool dilated_conv = false;
  memory::dims _stride(kdim), _dilation(kdim), _padding(kdim), _padding_r(kdim);
  for (size_t d = 0; d < kdim; ++d) {
    _stride[d] = stride[d];
    _dilation[d] = dilation[d];
    _padding[d] = padding[d];

    if (dilation[d] != 1) dilated_conv = true;

    auto osize = output.size(d + 2);
    auto isize = input.size(d + 2);
    auto ksize = weight.size(d + 2);
    _padding_r[d] = (osize - 1) * stride[d] - isize + ((ksize - 1) * dilation[d] + 1) - padding[d];
  }

  auto data_t = memory::data_type::f32;
  auto format_any = memory::format::any;
  auto format_x = memory::format::x;

  memory::dims input_tz, weight_tz, bias_tz, output_tz;
  memory::format format_data, format_weight;
  if (dim == 4) {
    input_tz = {n, ic, ih, iw};
    weight_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};
    bias_tz = {oc};
    output_tz = {n, oc, oh, ow};
    format_data = memory::format::nchw;
    format_weight = (g!= 1) ? memory::format::goihw : memory::format::oihw;
  } else {
    input_tz = {n, ic, id, ih, iw};
    weight_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kd, kh, kw} : memory::dims{oc, ic, kd, kh, kw};
    bias_tz = {oc};
    output_tz = {n, oc, od, oh, ow};
    format_data = memory::format::ncdhw;
    format_weight = (g!= 1) ? memory::format::goidhw : memory::format::oidhw;
  }

  auto input_md = memory::desc({input_tz}, data_t, format_any);
  auto weight_md = memory::desc({weight_tz}, data_t, format_any);
  auto bias_md = memory::desc({bias_tz}, data_t, format_any);
  auto output_md = memory::desc({output_tz}, data_t, format_any);

  std::shared_ptr<convolution_forward::desc> conv_forward_desc;
  if (bias.defined()) {
    if (dilated_conv) {
      conv_forward_desc.reset(new convolution_forward::desc(
        conv_prop, conv_algo, input_md, weight_md, bias_md, output_md,
        _stride, _dilation, _padding, _padding_r, padding_kind::zero));
    } else {
      conv_forward_desc.reset(new convolution_forward::desc(
        conv_prop, conv_algo, input_md, weight_md, bias_md, output_md,
        _stride, _padding, _padding, padding_kind::zero));
    }
  } else {
    if (dilated_conv) {
      conv_forward_desc.reset(new convolution_forward::desc(
        conv_prop, conv_algo, input_md, weight_md, output_md,
        _stride, _dilation, _padding, _padding_r, padding_kind::zero));
    } else {
      conv_forward_desc.reset(new convolution_forward::desc(
        conv_prop, conv_algo, input_md, weight_md, output_md,
        _stride, _padding, _padding, padding_kind::zero));
    }
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  conv_forward_pd.reset(new convolution_forward::primitive_desc(
    *conv_forward_desc, cpu_engine));

  auto input_usr_memory = memory({{{input_tz}, data_t, format_data}, cpu_engine},
    input.data_ptr());
  auto weight_usr_memory = memory({{{weight_tz}, data_t,  format_weight}, cpu_engine},
    weight.data_ptr());
  auto output_usr_memory = memory({{{output_tz}, data_t, format_data}, cpu_engine},
    output.data_ptr());

  std::vector<primitive> net;

  auto input_pd = conv_forward_pd->src_primitive_desc();
  auto input_memory = input_usr_memory;
  if (input_usr_memory.get_primitive_desc() != memory::primitive_desc(input_pd)) {
    input_memory = memory(input_pd);
    net.push_back(reorder(input_usr_memory, input_memory));
  }

  auto weight_pd = conv_forward_pd->weights_primitive_desc();
  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_primitive_desc() != memory::primitive_desc(weight_pd)) {
    weight_memory = memory(weight_pd);
    net.push_back(reorder(weight_usr_memory, weight_memory));
  }

  auto output_pd = conv_forward_pd->dst_primitive_desc();
  auto output_memory = output_usr_memory;
  if (output_usr_memory.get_primitive_desc() != memory::primitive_desc(output_pd)) {
    output_memory = memory(output_pd);
  }

  std::shared_ptr<convolution_forward> conv_forward;
  std::shared_ptr<memory> bias_usr_memory;
  if (bias.defined()) {
    bias_usr_memory.reset(new memory({{{bias_tz}, data_t, format_x}, cpu_engine},
      bias.data_ptr()));
    conv_forward.reset(new convolution_forward(*conv_forward_pd, input_memory,
      weight_memory, *bias_usr_memory, output_memory));
  } else {
    conv_forward.reset(new convolution_forward(*conv_forward_pd, input_memory,
      weight_memory, output_memory));
  }
  net.push_back(*conv_forward);

  if (output_memory != output_usr_memory) {
    net.push_back(reorder(output_memory, output_usr_memory));
  }

  Stream::Instance().get_stream().submit(net);

  return output;
}

Tensor mkldnn_convolution_backward_input(
    IntList input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation, int64_t groups, bool bias_defined)
{
  auto grad_input = at::empty(input_size, grad_output.options());

  auto cpu_engine = CpuEngine::Instance().get_engine();
  // TODO: winograd?
  auto conv_algo = algorithm::convolution_direct;

  auto dim = grad_output.dim();
  auto kdim = dim - 2;
  auto depth_dim = 2;
  auto height_dim = (dim == 5) ? 3 : 2;
  auto weight_dim = (dim == 5) ? 4 : 3;

  int32_t g = groups;

  int32_t n = grad_input.size(0);
  int32_t ic = grad_input.size(1);
  int32_t id = grad_input.size(depth_dim);
  int32_t ih = grad_input.size(height_dim);
  int32_t iw = grad_input.size(weight_dim);

  int32_t oc = grad_output.size(1);
  int32_t od = grad_output.size(depth_dim);
  int32_t oh = grad_output.size(height_dim);
  int32_t ow = grad_output.size(weight_dim);

  int32_t kd = weight.size(depth_dim);
  int32_t kh = weight.size(height_dim);
  int32_t kw = weight.size(weight_dim);

  bool dilated_conv = false;
  memory::dims _stride(kdim), _dilation(kdim), _padding(kdim), _padding_r(kdim);
  for (size_t d = 0; d < kdim; ++d) {
    _stride[d] = stride[d];
    _dilation[d] = dilation[d];
    _padding[d] = padding[d];

    if (dilation[d] != 1) dilated_conv = true;

    auto osize = grad_output.size(d + 2);
    auto isize = grad_input.size(d + 2);
    auto ksize = weight.size(d + 2);
    _padding_r[d] = (osize - 1) * stride[d] - isize + ((ksize - 1) * dilation[d] + 1) - padding[d];
  }

  auto data_t = memory::data_type::f32;
  auto format_any = memory::format::any;
  auto format_x = memory::format::x;

  memory::dims input_tz, weight_tz, bias_tz, output_tz;
  memory::format format_data, format_weight;
  if (dim == 4) {
    input_tz = {n, ic, ih, iw};
    weight_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};
    bias_tz = {oc};
    output_tz = {n, oc, oh, ow};
    format_data = memory::format::nchw;
    format_weight = (g!= 1) ? memory::format::goihw : memory::format::oihw;
  } else {
    input_tz = {n, ic, id, ih, iw};
    weight_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kd, kh, kw} : memory::dims{oc, ic, kd, kh, kw};
    bias_tz = {oc};
    output_tz = {n, oc, od, oh, ow};
    format_data = memory::format::ncdhw;
    format_weight = (g!= 1) ? memory::format::goidhw : memory::format::oidhw;
  }

  auto input_md = memory::desc({input_tz}, data_t, format_any);
  auto weight_md = memory::desc({weight_tz}, data_t, format_any);
  auto bias_md = memory::desc({bias_tz}, data_t, format_any);
  auto output_md = memory::desc({output_tz}, data_t, format_any);

  // need to re-create conv_forward_pd to feed conv_backward_data_pd
  // TODO: cache conv_forward_pd
  std::shared_ptr<convolution_forward::desc> conv_forward_desc;
  if (bias_defined) {
    if (dilated_conv) {
      conv_forward_desc.reset(new convolution_forward::desc(
        prop_kind::forward, conv_algo, input_md, weight_md, bias_md, output_md,
        _stride, _dilation, _padding, _padding_r, padding_kind::zero));
    } else {
      conv_forward_desc.reset(new convolution_forward::desc(
        prop_kind::forward, conv_algo, input_md, weight_md, bias_md, output_md,
        _stride, _padding, _padding, padding_kind::zero));
    }
  } else {
    if (dilated_conv) {
      conv_forward_desc.reset(new convolution_forward::desc(
        prop_kind::forward, conv_algo, input_md, weight_md, output_md,
        _stride, _dilation, _padding, _padding_r, padding_kind::zero));
    } else {
      conv_forward_desc.reset(new convolution_forward::desc(
        prop_kind::forward, conv_algo, input_md, weight_md, output_md,
        _stride, _padding, _padding, padding_kind::zero));
    }
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  conv_forward_pd.reset(new convolution_forward::primitive_desc(
    *conv_forward_desc, cpu_engine));

  std::shared_ptr<convolution_backward_data::desc> conv_backward_data_desc;
  if (dilated_conv) {
    conv_backward_data_desc.reset(new convolution_backward_data::desc(
      conv_algo, input_md, weight_md, output_md,
      _stride, _dilation, _padding, _padding_r, padding_kind::zero));
  } else {
    conv_backward_data_desc.reset(new convolution_backward_data::desc(
      conv_algo, input_md, weight_md, output_md,
      _stride, _padding, _padding, padding_kind::zero));
  }

  std::shared_ptr<convolution_backward_data::primitive_desc> conv_backward_data_pd;
  conv_backward_data_pd.reset(new convolution_backward_data::primitive_desc(
    *conv_backward_data_desc, cpu_engine, *conv_forward_pd));

  auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_data}, cpu_engine},
    grad_output.data_ptr());
  auto weight_usr_memory = memory({{{weight_tz}, data_t, format_weight}, cpu_engine},
    weight.data_ptr());
  auto grad_input_usr_memory = memory({{{input_tz}, data_t, format_data}, cpu_engine},
    grad_input.data_ptr());

  std::vector<primitive> net;

  auto grad_output_pd = conv_backward_data_pd->diff_dst_primitive_desc();
  auto grad_output_memory = grad_output_usr_memory;
  if (grad_output_usr_memory.get_primitive_desc() != memory::primitive_desc(grad_output_pd)) {
    grad_output_memory = memory(grad_output_pd);
    net.push_back(reorder(grad_output_usr_memory, grad_output_memory));
  }

  auto weight_pd = conv_backward_data_pd->weights_primitive_desc();
  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_primitive_desc() != memory::primitive_desc(weight_pd)) {
    weight_memory = memory(weight_pd);
    net.push_back(reorder(weight_usr_memory, weight_memory));
  }

  auto grad_input_pd = conv_backward_data_pd->diff_src_primitive_desc();
  auto grad_input_memory = grad_input_usr_memory;
  if (grad_input_memory.get_primitive_desc() != memory::primitive_desc(grad_input_pd)) {
    grad_input_memory = memory(grad_input_pd);
  }

  std::shared_ptr<convolution_backward_data> conv_backward_data;
  conv_backward_data.reset(new convolution_backward_data(*conv_backward_data_pd,
    grad_output_memory, weight_memory, grad_input_memory));
  net.push_back(*conv_backward_data);

  if (grad_input_memory != grad_input_usr_memory) {
    net.push_back(reorder(grad_input_memory, grad_input_usr_memory));
  }

  Stream::Instance().get_stream().submit(net);

  return grad_input;
}

std::tuple<at::Tensor, at::Tensor> mkldnn_convolution_backward_weights(
    IntList weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntList padding, IntList stride, IntList dilation, int64_t groups, bool bias_defined)
{
  auto grad_weight = at::empty(weight_size, grad_output.options());

  Tensor grad_bias;
  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
  }

  auto cpu_engine = CpuEngine::Instance().get_engine();
  // TODO: winograd?
  auto conv_algo = algorithm::convolution_direct;

  auto dim = grad_output.dim();
  auto kdim = dim - 2;
  auto depth_dim = 2;
  auto height_dim = (dim == 5) ? 3 : 2;
  auto weight_dim = (dim == 5) ? 4 : 3;

  int32_t g = groups;

  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t id = input.size(depth_dim);
  int32_t ih = input.size(height_dim);
  int32_t iw = input.size(weight_dim);

  int32_t oc = grad_output.size(1);
  int32_t od = grad_output.size(depth_dim);
  int32_t oh = grad_output.size(height_dim);
  int32_t ow = grad_output.size(weight_dim);

  int32_t kd = grad_weight.size(depth_dim);
  int32_t kh = grad_weight.size(height_dim);
  int32_t kw = grad_weight.size(weight_dim);

  bool dilated_conv = false;
  memory::dims _stride(kdim), _dilation(kdim), _padding(kdim), _padding_r(kdim);
  for (size_t d = 0; d < kdim; ++d) {
    _stride[d] = stride[d];
    _dilation[d] = dilation[d];
    _padding[d] = padding[d];

    if (dilation[d] != 1) dilated_conv = true;

    auto osize = grad_output.size(d + 2);
    auto isize = input.size(d + 2);
    auto ksize = grad_weight.size(d + 2);
    _padding_r[d] = (osize - 1) * stride[d] - isize + ((ksize - 1) * dilation[d] + 1) - padding[d];
  }

  auto data_t = memory::data_type::f32;
  auto format_any = memory::format::any;
  auto format_x = memory::format::x;

  memory::dims input_tz, weight_tz, bias_tz, output_tz;
  memory::format format_data, format_weight;
  if (dim == 4) {
    input_tz = {n, ic, ih, iw};
    weight_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};
    bias_tz = {oc};
    output_tz = {n, oc, oh, ow};
    format_data = memory::format::nchw;
    format_weight = (g!= 1) ? memory::format::goihw : memory::format::oihw;
  } else {
    input_tz = {n, ic, id, ih, iw};
    weight_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kd, kh, kw} : memory::dims{oc, ic, kd, kh, kw};
    bias_tz = {oc};
    output_tz = {n, oc, od, oh, ow};
    format_data = memory::format::ncdhw;
    format_weight = (g!= 1) ? memory::format::goidhw : memory::format::oidhw;
  }

  auto input_md = memory::desc({input_tz}, data_t, format_any);
  auto weight_md = memory::desc({weight_tz}, data_t, format_any);
  auto bias_md = memory::desc({bias_tz}, data_t, format_any);
  auto output_md = memory::desc({output_tz}, data_t, format_any);

  // need to re-create conv_forward_pd to feed conv_backward_weight_pd
  // TODO: cache conv_forward_pd
  std::shared_ptr<convolution_forward::desc> conv_forward_desc;
  if (bias_defined) {
    if (dilated_conv) {
      conv_forward_desc.reset(new convolution_forward::desc(
        prop_kind::forward, conv_algo, input_md, weight_md, bias_md, output_md,
        _stride, _dilation, _padding, _padding_r, padding_kind::zero));
    } else {
      conv_forward_desc.reset(new convolution_forward::desc(
        prop_kind::forward, conv_algo, input_md, weight_md, bias_md, output_md,
        _stride, _padding, _padding, padding_kind::zero));
    }
  } else {
    if (dilated_conv) {
      conv_forward_desc.reset(new convolution_forward::desc(
        prop_kind::forward, conv_algo, input_md, weight_md, output_md,
        _stride, _dilation, _padding, _padding_r, padding_kind::zero));
    } else {
      conv_forward_desc.reset(new convolution_forward::desc(
        prop_kind::forward, conv_algo, input_md, weight_md, output_md,
        _stride, _padding, _padding, padding_kind::zero));
    }
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  conv_forward_pd.reset(new convolution_forward::primitive_desc(
    *conv_forward_desc, cpu_engine));

  std::shared_ptr<convolution_backward_weights::desc> conv_backward_weight_desc;
  if (bias_defined) {
    if (dilated_conv) {
      conv_backward_weight_desc.reset(new convolution_backward_weights::desc(
        conv_algo, input_md, weight_md, bias_md, output_md,
        _stride, _dilation, _padding, _padding_r, padding_kind::zero));
    } else {
      conv_backward_weight_desc.reset(new convolution_backward_weights::desc(
        conv_algo, input_md, weight_md, bias_md, output_md,
        _stride, _padding, _padding, padding_kind::zero));
    }
  } else {
    if (dilated_conv) {
      conv_backward_weight_desc.reset(new convolution_backward_weights::desc(
        conv_algo, input_md, weight_md, output_md,
        _stride, _dilation, _padding, _padding_r, padding_kind::zero));
    } else {
      conv_backward_weight_desc.reset(new convolution_backward_weights::desc(
        conv_algo, input_md, weight_md, output_md,
        _stride, _padding, _padding, padding_kind::zero));
    }
  }

  std::shared_ptr<convolution_backward_weights::primitive_desc> conv_backward_weight_pd;
  conv_backward_weight_pd.reset(new convolution_backward_weights::primitive_desc(
    *conv_backward_weight_desc, cpu_engine, *conv_forward_pd));

  auto input_usr_memory = memory({{{input_tz}, data_t, format_data}, cpu_engine},
    input.data_ptr());
  auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_data}, cpu_engine},
    grad_output.data_ptr());
  auto grad_weight_usr_memory = memory({{{weight_tz}, data_t, format_weight}, cpu_engine},
    grad_weight.data_ptr());
  std::shared_ptr<memory> grad_bias_memory;

  std::vector<primitive> net;

  auto input_pd = conv_backward_weight_pd->src_primitive_desc();
  auto input_memory = input_usr_memory;
  if (input_usr_memory.get_primitive_desc() != memory::primitive_desc(input_pd)) {
    input_memory = memory(input_pd);
    net.push_back(reorder(input_usr_memory, input_memory));
  }

  auto grad_output_pd = conv_backward_weight_pd->diff_dst_primitive_desc();
  auto grad_output_memory = grad_output_usr_memory;
  if (grad_output_usr_memory.get_primitive_desc() != memory::primitive_desc(grad_output_pd)) {
    grad_output_memory = memory(grad_output_pd);
    net.push_back(reorder(grad_output_usr_memory, grad_output_memory));
  }

  auto grad_weight_pd = conv_backward_weight_pd->diff_weights_primitive_desc();
  auto grad_weight_memory = grad_weight_usr_memory;
  if (grad_weight_usr_memory.get_primitive_desc() != memory::primitive_desc(grad_weight_pd)) {
    grad_weight_memory = memory(grad_weight_pd);
  }

  std::shared_ptr<convolution_backward_weights> conv_backward_weight;
  if (bias_defined) {
    grad_bias_memory.reset(new memory({{{bias_tz}, data_t, format_x}, cpu_engine},
      grad_bias.data_ptr()));
    conv_backward_weight.reset(new convolution_backward_weights(*conv_backward_weight_pd,
      input_memory, grad_output_memory, grad_weight_memory, *grad_bias_memory));
  } else {
    conv_backward_weight.reset(new convolution_backward_weights(*conv_backward_weight_pd,
      input_memory, grad_output_memory, grad_weight_memory));
  }

  net.push_back(*conv_backward_weight);

  if (grad_weight_memory != grad_weight_usr_memory) {
    net.push_back(reorder(grad_weight_memory, grad_weight_usr_memory));
  }

  Stream::Instance().get_stream().submit(net);

  return std::tuple<at::Tensor, at::Tensor>{grad_weight, grad_bias};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation, int64_t groups, std::array<bool,3> output_mask)
{
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

}}  // namespace at::native

#endif

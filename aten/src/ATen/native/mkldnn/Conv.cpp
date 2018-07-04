#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

at::Tensor mkldnn_convolution(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    IntList padding, IntList stride, IntList dilation) {
  throw std::runtime_error("mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

at::Tensor mkldnn_convolution_backward_input(
    IntList input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation, bool bias_defined) {
  throw std::runtime_error("mkldnn_convolution_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<at::Tensor,at::Tensor> mkldnn_convolution_backward_weights(
    IntList weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntList padding, IntList stride, IntList dilation, bool bias_defined) {
  throw std::runtime_error("mkldnn_convolution_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation, std::array<bool,3> output_mask) {
  throw std::runtime_error("mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
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

std::vector<int64_t> conv_output_size(
    IntList input_size, IntList weight_size,
    IntList padding, IntList stride, IntList dilation)
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
    IntList padding, IntList stride, IntList dilation)
{
  auto output = input.type().tensor(conv_output_size(
    input.sizes(), weight.sizes(), padding, stride, dilation));

  auto cpu_engine = CpuEngine::Instance().get_engine();
  
  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t ih = input.size(2);
  int32_t iw = input.size(3);

  int32_t oc = output.size(1);
  int32_t oh = output.size(2);
  int32_t ow = output.size(3);

  int32_t kh = weight.size(2);
  int32_t kw = weight.size(3);

  int32_t sh = stride[0];
  int32_t sw = stride[1];
  int32_t ph = padding[0];
  int32_t pw = padding[1];

  auto data_t = memory::data_type::f32;
  auto format_any = memory::format::any;
  auto format_nchw = memory::format::nchw;
  auto format_oihw = memory::format::oihw;
  auto format_x = memory::format::x;

  memory::dims input_tz = {n, ic, ih, iw};
  memory::dims weight_tz = {oc, ic, kh, kw};
  memory::dims bias_tz = {oc};
  memory::dims output_tz = {n, oc, oh, ow};
  memory::dims _stride = {sh, sw};
  memory::dims _padding = {ph, pw};

  auto input_md = memory::desc({input_tz}, data_t, format_any);
  auto weight_md = memory::desc({weight_tz}, data_t, format_any);
  auto bias_md = memory::desc({bias_tz}, data_t, format_any);
  auto output_md = memory::desc({output_tz}, data_t, format_any);

  std::shared_ptr<convolution_forward::desc> conv_forward_desc;
  if (bias.defined()) {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      convolution_direct, input_md, weight_md, bias_md, output_md,
      _stride, _padding, _padding, padding_kind::zero));
  } else {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      convolution_direct, input_md, weight_md, output_md,
      _stride, _padding, _padding, padding_kind::zero));
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  conv_forward_pd.reset(new convolution_forward::primitive_desc(
    *conv_forward_desc, cpu_engine));

  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, cpu_engine},
    input.data_ptr());
  auto weight_usr_memory = memory({{{weight_tz}, data_t,  format_oihw}, cpu_engine},
    weight.data_ptr());
  auto output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, cpu_engine},
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
    IntList padding, IntList stride, IntList dilation, bool bias_defined)
{
  auto grad_input = grad_output.type().tensor(input_size);

  auto cpu_engine = CpuEngine::Instance().get_engine();

  int32_t n = grad_input.size(0);
  int32_t ic = grad_input.size(1);
  int32_t ih = grad_input.size(2);
  int32_t iw = grad_input.size(3);

  int32_t oc = grad_output.size(1);
  int32_t oh = grad_output.size(2);
  int32_t ow = grad_output.size(3);

  int32_t kh = weight.size(2);
  int32_t kw = weight.size(3);

  int32_t sh = stride[0];
  int32_t sw = stride[1];
  int32_t ph = padding[0];
  int32_t pw = padding[1];

  auto data_t = memory::data_type::f32;
  auto format_any = memory::format::any;
  auto format_nchw = memory::format::nchw;
  auto format_oihw = memory::format::oihw;

  memory::dims input_tz = {n, ic, ih, iw};
  memory::dims weight_tz = {oc, ic, kh, kw};
  memory::dims bias_tz = {oc};
  memory::dims output_tz = {n, oc, oh, ow};
  memory::dims _stride = {sh, sw};
  memory::dims _padding = {ph, pw};

  auto input_md = memory::desc({input_tz}, data_t, format_any);
  auto weight_md = memory::desc({weight_tz}, data_t, format_any);
  auto bias_md = memory::desc({bias_tz}, data_t, format_any);
  auto output_md = memory::desc({output_tz}, data_t, format_any);

  // need to re-create conv_forward_pd to feed conv_backward_data_pd
  std::shared_ptr<convolution_forward::desc> conv_forward_desc;
  if (bias_defined) {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      convolution_direct, input_md, weight_md, bias_md, output_md,
      _stride, _padding, _padding, padding_kind::zero));
  } else {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      convolution_direct, input_md, weight_md, output_md,
      _stride, _padding, _padding, padding_kind::zero));
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  conv_forward_pd.reset(new convolution_forward::primitive_desc(
    *conv_forward_desc, cpu_engine));

  std::shared_ptr<convolution_backward_data::desc> conv_backward_data_desc;
  conv_backward_data_desc.reset(new convolution_backward_data::desc(
    convolution_direct, input_md, weight_md, output_md,
    _stride, _padding, _padding, padding_kind::zero));

  std::shared_ptr<convolution_backward_data::primitive_desc> conv_backward_data_pd;
  conv_backward_data_pd.reset(new convolution_backward_data::primitive_desc(
    *conv_backward_data_desc, cpu_engine, *conv_forward_pd));

  auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, cpu_engine},
    grad_output.data_ptr());
  auto weight_usr_memory = memory({{{weight_tz}, data_t, format_oihw}, cpu_engine},
    weight.data_ptr());
  auto grad_input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, cpu_engine},
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
    IntList padding, IntList stride, IntList dilation, bool bias_defined)
{
  auto grad_weight = grad_output.type().tensor(weight_size);

  Tensor grad_bias;
  if (bias_defined) {
    grad_bias = grad_output.type().tensor({grad_output.size(1)});
  }

  auto cpu_engine = CpuEngine::Instance().get_engine();

  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t ih = input.size(2);
  int32_t iw = input.size(3);

  int32_t oc = grad_output.size(1);
  int32_t oh = grad_output.size(2);
  int32_t ow = grad_output.size(3);

  int32_t kh = grad_weight.size(2);
  int32_t kw = grad_weight.size(3);

  int32_t sh = stride[0];
  int32_t sw = stride[1];
  int32_t ph = padding[0];
  int32_t pw = padding[1];

  auto data_t = memory::data_type::f32;
  auto format_any = memory::format::any;
  auto format_nchw = memory::format::nchw;
  auto format_oihw = memory::format::oihw;
  auto format_x = memory::format::x;

  memory::dims input_tz = {n, ic, ih, iw};
  memory::dims weight_tz = {oc, ic, kh, kw};
  memory::dims bias_tz = {oc};
  memory::dims output_tz = {n, oc, oh, ow};
  memory::dims _stride = {sh, sw};
  memory::dims _padding = {ph, pw};

  memory::desc input_md({input_tz}, data_t, format_any);
  memory::desc weight_md({weight_tz}, data_t, format_any);
  memory::desc bias_md({bias_tz}, data_t, format_any);
  memory::desc output_md({output_tz}, data_t, format_any);

  // need to re-create conv_forward_pd to feed conv_backward_weight_pd
  std::shared_ptr<convolution_forward::desc> conv_forward_desc;
  if (bias_defined) {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      convolution_direct, input_md, weight_md, bias_md, output_md,
      _stride, _padding, _padding, padding_kind::zero));
  } else {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      convolution_direct, input_md, weight_md, output_md,
      _stride, _padding, _padding, padding_kind::zero));
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  conv_forward_pd.reset(new convolution_forward::primitive_desc(
    *conv_forward_desc, cpu_engine));

  std::shared_ptr<convolution_backward_weights::desc> conv_backward_weight_desc;
  if (bias_defined) {
    conv_backward_weight_desc.reset(new convolution_backward_weights::desc(
      convolution_direct, input_md, weight_md, bias_md, output_md,
      _stride, _padding, _padding, padding_kind::zero));
  } else {
    conv_backward_weight_desc.reset(new convolution_backward_weights::desc(
      convolution_direct, input_md, weight_md, output_md,
      _stride, _padding, _padding, padding_kind::zero));
  }

  std::shared_ptr<convolution_backward_weights::primitive_desc> conv_backward_weight_pd;
  conv_backward_weight_pd.reset(new convolution_backward_weights::primitive_desc(
    *conv_backward_weight_desc, cpu_engine, *conv_forward_pd));

  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, cpu_engine},
    input.data_ptr());
  auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, cpu_engine},
    grad_output.data_ptr());
  auto grad_weight_usr_memory = memory({{{weight_tz}, data_t, format_oihw}, cpu_engine},
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
    IntList padding, IntList stride, IntList dilation, std::array<bool,3> output_mask)
{
  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding, stride, dilation, output_mask[2]);
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

}}  // namespace at::native

#endif

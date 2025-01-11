#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/onednn/MKLDNNCommon.h>
#include <ATen/native/onednn/Utils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/mkldnn_reorder_conv2d_weight_native.h>
#include <ATen/ops/mkldnn_reorder_conv3d_weight_native.h>
#include <ATen/ops/to_mkldnn_native.h>
#include <ATen/ops/zeros.h>
#endif


namespace at::native {

#if AT_MKLDNN_ENABLED()

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor, std::optional<ScalarType> dtype, std::optional<bool> masked_grad) {
  TORCH_CHECK(mkldnn_tensor.scalar_type() == ScalarType::Float ||
              mkldnn_tensor.scalar_type() == ScalarType::BFloat16 ||
              mkldnn_tensor.scalar_type() == ScalarType::Half ||
              mkldnn_tensor.scalar_type() == ScalarType::Byte ||
              mkldnn_tensor.scalar_type() == ScalarType::Char,
              "mkldnn_to_dense expects float, bfloat16, half, uint8, int8 tensor input");
  ideep::tensor& stensor = itensor_from_mkldnn(mkldnn_tensor);
  auto dims = stensor.get_dims();
  auto data_type = dtype.has_value() ? dtype.value() : mkldnn_tensor.scalar_type();
  TORCH_CHECK(data_type == ScalarType::Float ||
              data_type == ScalarType::BFloat16 ||
              data_type == ScalarType::Half ||
              data_type == ScalarType::Byte ||
              data_type == ScalarType::Char,
              "mkldnn tensor only can be converted to be a float, bfloat16, Half, uint8, int8 cpu tensor")
  if (mkldnn_tensor.scalar_type() == ScalarType::Byte || mkldnn_tensor.scalar_type() == ScalarType::Char) {
    // For int8, uint8 input, we should not change the data type.
    TORCH_CHECK(mkldnn_tensor.scalar_type() == data_type,
            "For int8, uint8 mkldnn_tensor input, we should not change the data type.");
  }
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  Tensor cpu_tensor = at::empty(
    std::vector<int64_t>(dims.begin(), dims.end()),
    mkldnn_tensor.options().layout(c10::kStrided).dtype(data_type));
  if (stensor.is_empty()) return cpu_tensor;
  auto pub_tensor =
      data_type == ScalarType::Float
      ? stensor.to_public(cpu_tensor.template data_ptr<float>(),
                          ideep::tensor::data_type::f32)
      : (data_type == ScalarType::BFloat16
         ? stensor.to_public(cpu_tensor.template data_ptr<BFloat16>(),
                         ideep::tensor::data_type::bf16)
         : (data_type == ScalarType::Half
            ? stensor.to_public(cpu_tensor.template data_ptr<Half>(),
                            ideep::tensor::data_type::f16)
          : (data_type == ScalarType::Byte
              ? stensor.to_public(cpu_tensor.template data_ptr<uint8_t>(),
                              ideep::tensor::data_type::u8)
              : stensor.to_public(cpu_tensor.template data_ptr<int8_t>(),
                              ideep::tensor::data_type::s8)
            )
           )
      );
  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  // Make sure that NC11 strides follow formula of contiguous tensor.
  return cpu_tensor.contiguous().resize_(dims, c10::MemoryFormat::Contiguous);
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor, std::optional<ScalarType> dtype) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(),
             "dense_to_mkldnn expects CPU tensor input");
  TORCH_CHECK(cpu_tensor.layout() == Layout::Strided,
             "dense_to_mkldnn expects strided tensor input");
  TORCH_CHECK(cpu_tensor.scalar_type() == ScalarType::Float ||
              cpu_tensor.scalar_type() == ScalarType::BFloat16 ||
              cpu_tensor.scalar_type() == ScalarType::Half ||
              cpu_tensor.scalar_type() == ScalarType::Byte ||
              cpu_tensor.scalar_type() == ScalarType::Char,
             "dense_to_mkldnn expects float, bfloat16, half, uint8, int8 tensor input");
  TORCH_CHECK(cpu_tensor.dim() <= 5,
             "Can't convert cpu tensor with the number of dimensions > 5");
  // NOTE: forbid direct convert from non-contiguous (or channels last) to `ideep::tensor`.
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  auto data_type = dtype.has_value() ? dtype.value() : cpu_tensor.scalar_type();
  if (cpu_tensor.scalar_type() == ScalarType::Byte || cpu_tensor.scalar_type() == ScalarType::Char) {
    // For int8, uint8 input, we should not change the data type.
    TORCH_CHECK(cpu_tensor.scalar_type() == data_type,
            "For int8, uint8 cpu_tensor input, we should not change the data type.");
  }
  TORCH_CHECK(data_type == ScalarType::Float ||
              data_type == ScalarType::BFloat16 ||
              data_type == ScalarType::Half ||
              data_type == ScalarType::Byte ||
              data_type == ScalarType::Char,
              "cpu tensor only can be converted to be a float, bfloat16, half, uint8, int8 mkldnn tensor")
  Tensor mkldnn_tensor = empty_mkldnn(cpu_tensor_cont.sizes(), data_type,
                                      cpu_tensor_cont.options().layout_opt(), cpu_tensor_cont.options().device_opt(),
                                      cpu_tensor_cont.options().pinned_memory_opt());
  ideep::tensor& dtensor = itensor_from_mkldnn(mkldnn_tensor);
  if (cpu_tensor.scalar_type() == ScalarType::Float) {
    dtensor.feed_from(dtensor.get_dims(),
                      ideep::tensor::data_type::f32,
                      (cpu_tensor_cont.template data_ptr<float>()));
  } else if (cpu_tensor.scalar_type() == ScalarType::BFloat16) {
    dtensor.feed_from(dtensor.get_dims(),
                      ideep::tensor::data_type::bf16,
                      cpu_tensor_cont.template data_ptr<BFloat16>());
  } else if (cpu_tensor.scalar_type() == ScalarType::Half) {
    dtensor.feed_from(dtensor.get_dims(),
                      ideep::tensor::data_type::f16,
                      cpu_tensor_cont.template data_ptr<Half>());
  } else if (cpu_tensor.scalar_type() == ScalarType::Byte) {
    dtensor.feed_from(dtensor.get_dims(),
                      ideep::tensor::data_type::u8,
                      cpu_tensor_cont.template data_ptr<uint8_t>());
  } else {
    TORCH_CHECK(cpu_tensor.scalar_type() == ScalarType::Char,
            "Expect int8 input of cpu_tensor");
    dtensor.feed_from(dtensor.get_dims(),
                      ideep::tensor::data_type::s8,
                      cpu_tensor_cont.template data_ptr<int8_t>());
  }
  return mkldnn_tensor;
}

// Mkldnn tensor has special non-public format for conv2d weights
// (dense_to_mkldnn only converts dense tensor to mkldnn tensor with
// public format). Ideep conv kernel will do implicit reorder if the
// weight is not already in this optimized format. By the time I'm
// writing this note, we are seeing ~20% perf cost of doing the
// on-the-fly reorder.
Tensor mkldnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::OptionalArrayRef<int64_t> input_size) {
  mkldnn_check_low_precision(self.scalar_type(), "mkldnn_reorder_conv2d_weight");
  const auto padding_expanded = expand_param_if_needed(padding, "padding", 2);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", 2);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", 2);

  ideep::dims src_dims = ideep::dims();
  bool is_channels_last = false;
  auto memory_format = at::MemoryFormat::Contiguous;
  if (input_size.has_value()) {
    src_dims = input_size.value().vec();
    // if has input size, we always use channels last.
    is_channels_last = true;
    memory_format = at::MemoryFormat::ChannelsLast;
  }

  auto self_ = self.is_mkldnn() ? self : self.contiguous(memory_format);
  auto w = itensor_from_tensor(self_);

  // Legacy mkldnn conv2d jitted module may contain a 5-d weight with an extra
  // dimension when groups > 1, having dimension [g, o/g, i, h, w] instead of
  // [o, i, h, w]. Ideally we should reorder the weight back in serialization.
  // For backward compatibility, we squash the first two dims (g * o/g) back to
  // its original form.
  if (w.ndims() == 5) {
    auto wdims = w.get_dims();
    w.reshape({wdims[0] * wdims[1], wdims[2], wdims[3], wdims[4]});
  }

  auto desc = ideep::convolution_forward::expected_weights_desc(
      w.get_dims(),
      w.get_data_type(),
      stride_expanded,
      padding_expanded,
      padding_expanded,
      dilation_expanded,
      groups,
      ideep::algorithm::convolution_direct,
      ideep::prop_kind::forward,
      w.get_data_type(),
      src_dims,
      ideep::attr_t(),
      is_channels_last);
  ideep::tensor result;
  result.init(desc);
  result.feed_from(w);

  return new_with_itensor_mkldnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor mkldnn_reorder_conv3d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::OptionalArrayRef<int64_t> input_size) {
  mkldnn_check_low_precision(self.scalar_type(), "mkldnn_reorder_conv3d_weight");
  const auto padding_expanded = expand_param_if_needed(padding, "padding", 3);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", 3);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", 3);

  ideep::dims src_dims = ideep::dims();
  bool is_channels_last = false;
  auto memory_format = at::MemoryFormat::Contiguous;
  if (input_size.has_value()) {
    src_dims = input_size.value().vec();
    // if has input size, we always use channels last.
    is_channels_last = true;
    memory_format = at::MemoryFormat::ChannelsLast3d;
  }

  auto self_ = self.is_mkldnn() ? self : self.contiguous(memory_format);
  auto w = itensor_from_tensor(self_);

  auto desc = ideep::convolution_forward::expected_weights_desc(
      w.get_dims(),
      w.get_data_type(),
      stride_expanded,
      padding_expanded,
      padding_expanded,
      dilation_expanded,
      groups,
      ideep::algorithm::convolution_direct,
      ideep::prop_kind::forward,
      w.get_data_type(),
      src_dims,
      ideep::attr_t(),
      is_channels_last);
  ideep::tensor result;
  result.init(desc);
  result.feed_from(w);

  return new_with_itensor_mkldnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()), self.options().device_opt());
}

static Tensor mkldnn_reorder_conv_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::OptionalArrayRef<int64_t> input_size) {
  TORCH_CHECK((self.dim() == 4 || self.dim() == 5), "mkldnn_reorder_conv_weight only supports conv2d and conv3d");
  if (self.dim() == 4) {
    return at::native::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation, groups, input_size);
  } else {
    return at::native::mkldnn_reorder_conv3d_weight(self, padding, stride, dilation, groups, input_size);
  }
}

static Tensor mkldnn_reorder_linear_weight(
    const Tensor& self,
    std::optional<int64_t> batch_size_opt) {
  mkldnn_check_low_precision(self.scalar_type(), "mkldnn_reorder_linear_weight");
  auto out_features = self.size(0);
  auto in_features = self.size(1);
  auto self_ = self.contiguous();
  auto w = itensor_from_tensor(self_);
  ideep::dims input_size;
  auto dtype = w.get_data_type();
  if (batch_size_opt.has_value()) {
    input_size = {batch_size_opt.value(), in_features};
  }
  auto packed_desc = ideep::inner_product_forward::expected_weights_desc(
      {out_features, in_features},
      input_size,
      /* weight dtype */ dtype,
      /* src dtype */ dtype);
  ideep::tensor result;
  result.init(packed_desc);
  result.feed_from(w);
  return new_with_itensor_mkldnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()), self.options().device_opt());
}

static ideep::tensor::desc get_conv_transpose_expected_weights_desc(
    const ideep::tensor::dims& weights_dims,
    ideep::tensor::data_type w_dtype,
    const ideep::tensor::dims& strides,
    const ideep::tensor::dims& padding_l,
    const ideep::tensor::dims& padding_r,
    const ideep::tensor::dims& dilates,
    int groups,
    bool channels_last,
    ideep::algorithm aalgorithm,
    ideep::data_type x_dtype,
    const ideep::dims& src_dims) {
  if (channels_last) {
    return ideep::convolution_transpose_forward::expected_weights_desc<true>(
        weights_dims,
        w_dtype,
        strides,
        padding_l,
        padding_r,
        dilates,
        groups,
        aalgorithm,
        ideep::prop_kind::forward,
        src_dims);
  } else {
    return ideep::convolution_transpose_forward::expected_weights_desc<false>(
        weights_dims,
        w_dtype,
        strides,
        padding_l,
        padding_r,
        dilates,
        groups,
        aalgorithm,
        ideep::prop_kind::forward,
        src_dims);
  }
}

static Tensor mkldnn_reorder_conv_transpose_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::OptionalArrayRef<int64_t> input_size) {
  TORCH_CHECK(
      (self.dim() == 4 || self.dim() == 5),
      "mkldnn_reorder_conv_transpose_weight only supports conv_transpose2d and conv_transpose3d");
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  mkldnn_check_low_precision(
      self.scalar_type(), "mkldnn_reorder_conv_transpose_weight");
  int64_t pdim = self.dim() - 2;
  const auto padding_expanded =
      expand_param_if_needed(padding, "padding", pdim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", pdim);
  const auto dilation_expanded =
      expand_param_if_needed(dilation, "dilation", pdim);
  const auto output_padding_expanded =
      expand_param_if_needed(output_padding, "output_padding", pdim);

  ideep::dims src_dims = ideep::dims();
  bool is_channels_last = false;
  auto memory_format = at::MemoryFormat::Contiguous;
  if (input_size.has_value()) {
    src_dims = input_size.value().vec();
    // if has input size, we always use channels last.
    is_channels_last = true;
    memory_format = self.dim() == 4 ? at::MemoryFormat::ChannelsLast
                                    : at::MemoryFormat::ChannelsLast3d;
  }

  auto self_ = self.contiguous(memory_format);
  ideep::tensor w = itensor_from_tensor(self_);

  auto expected_desc = get_conv_transpose_expected_weights_desc(
      w.get_dims(),
      w.get_data_type(),
      stride_expanded,
      padding_expanded,
      padding_r(padding_expanded, output_padding_expanded),
      dilation_expanded,
      groups,
      is_channels_last,
      ideep::algorithm::deconvolution_direct,
      w.get_data_type(),
      src_dims);

  if (groups > 1) {
    expected_desc = expected_desc.transpose(1, 2);
  } else {
    expected_desc = expected_desc.transpose(0, 1);
  }

  ideep::tensor result;
  result.init(expected_desc);
  w.transpose_(0, 1);
  result.feed_from(w, /*is_deconv_weights*/true);

  return new_with_itensor_mkldnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

static std::tuple<ideep::tensor, ideep::tensor> get_lstm_packed_weights(
    const at::Tensor& weight_ih,
    const at::Tensor& weight_hh,
    const at::Tensor& weight2,
    const at::Tensor& weight3,
    int64_t layer_feature_size,
    int64_t hidden_size,
    bool has_biases,
    int64_t num_layers,
    bool bidirectional,
    int64_t time_step,
    int64_t batch_size,
    bool reverse) {

  ideep::tensor cached_weight_ih, cached_weight_hh;

  int64_t num_gates = 4;
  int64_t num_bias_gates = 4;
  std::vector<int64_t> output_sizes = {time_step, batch_size, hidden_size};

  auto dtype = get_mkldnn_dtype(weight_ih.scalar_type());
  ideep::tensor::desc src_layer_desc({time_step, batch_size, layer_feature_size}, dtype, ideep::format_tag::tnc);
  ideep::tensor::desc src_iter_desc({1, 1, batch_size, hidden_size}, dtype, ideep::format_tag::ldnc);
  ideep::tensor::desc src_iter_c_desc({1, 1, batch_size, hidden_size}, dtype, ideep::format_tag::ldnc);
  ideep::tensor::desc bias_desc({1, 1, num_bias_gates, hidden_size}, dtype, ideep::format_tag::ldgo);

  ideep::tensor::desc dst_layer_desc({time_step, batch_size, hidden_size}, dtype, ideep::format_tag::tnc);
  ideep::tensor::desc dst_iter_desc({1, 1, batch_size, hidden_size}, dtype, ideep::format_tag::ldnc);
  ideep::tensor::desc dst_iter_c_desc({1, 1, batch_size, hidden_size}, dtype, ideep::format_tag::ldnc);

  ideep::tensor src_layer(src_layer_desc);
  ideep::tensor src_iter(src_iter_desc);
  ideep::tensor src_iter_c(src_iter_c_desc);
  ideep::tensor bias(bias_desc);

  auto w1 = itensor_view_from_dense(
      weight_ih,
      {{1, 1, layer_feature_size, num_gates, hidden_size},
        get_mkldnn_dtype(weight_ih.scalar_type()),
        ideep::format_tag::ldgoi});

  auto w2 = itensor_view_from_dense(
      weight_hh,
      {{1, 1, hidden_size, num_gates, hidden_size},
        get_mkldnn_dtype(weight_hh.scalar_type()),
        ideep::format_tag::ldgoi});

  auto [packed_desc_ih, packed_desc_hh] =
      ideep::lstm_forward_inference::expected_weights_desc(
          output_sizes,
          src_layer,
          src_iter,
          src_iter_c,
          w1,
          w2,
          bias,
          reverse);

  cached_weight_ih.init(packed_desc_ih);
  cached_weight_hh.init(packed_desc_hh);

  cached_weight_ih.feed_from(w1);
  cached_weight_hh.feed_from(w2);

  return std::make_tuple(cached_weight_ih, cached_weight_hh);
}

static bool should_use_plain_format(ideep::tensor w) {
#if defined(IDEEP_VERSION_MAJOR) && IDEEP_VERSION_MAJOR>=3
  return w.get_desc().is_opaque() || w.get_desc().is_plain();
# else
  return w.get_desc().is_rnn_packed() || w.get_desc().is_plain();
#endif
}

static std::vector<Tensor> mkldnn_reorder_mkldnn_rnn_layer_weight(
 Tensor weight0,
 Tensor weight1,
 int64_t hidden_size,
 bool reverse,
 bool has_biases,
 bool batch_first,
 c10::OptionalArrayRef<int64_t> input_size) {

  std::vector<int64_t> input_size_value;
  int64_t time_step, batch_size;
  if (input_size.has_value()) {
    input_size_value = input_size.value().vec();
    int64_t time_index = batch_first ? 1: 0;
    int64_t batch_size_index = batch_first ? 0: 1;

    time_step = input_size_value[time_index];
    batch_size = input_size_value[batch_size_index];
  } else {
    // no value fed, provide one here
    time_step = 5;
    batch_size = 10;
  }

  at::Tensor packed_w1, packed_w2;

  int64_t feature_size = weight0.size(-1);

  auto [w1_, w2_] = get_lstm_packed_weights(
    weight0,
    weight1,
    at::zeros(
      weight0.sizes(),
      weight0.options()),
    at::zeros(
      weight1.sizes(),
      weight1.options()),
    feature_size,
    hidden_size,
    has_biases, // has_biases
    1, // num_layers
    false, // bidirectional
    time_step,
    batch_size,
    reverse);

  if (should_use_plain_format(w1_)) {
    packed_w1 = weight0;
  } else {
    packed_w1 = new_with_itensor_mkldnn(std::move(w1_), optTypeMetaToScalarType(weight0.options().dtype_opt()), weight0.options().device_opt());
  }

  if (should_use_plain_format(w2_)) {
    packed_w2 = weight1;
  } else {
    packed_w2 = new_with_itensor_mkldnn(std::move(w2_), optTypeMetaToScalarType(weight1.options().dtype_opt()), weight1.options().device_opt());
  }
  return {packed_w1, packed_w2};
}

static Tensor get_mkldnn_serialized_md(const Tensor& self) {
  const ideep::tensor packed_w = itensor_from_tensor(self);
  auto packed_w_desc = packed_w.get_desc();
  std::vector<uint8_t> serialized_wei_desc;

#if IDEEP_PREREQ(3, 4, 1, 2)
  serialized_wei_desc = packed_w_desc.get_blob();
#else
      TORCH_CHECK(false, "Unexpected IDeep version to do weight serialization.");
#endif
  Tensor serialized_md = at::from_blob((void*)serialized_wei_desc.data(), {(int64_t)serialized_wei_desc.size()}, at::TensorOptions(at::kByte));
  auto res = at::empty_like(serialized_md);
  // serialized_md shares the buffer with serialized_wei_desc,
  // which will be released outside of this function thus invalidating the buffer of serialized_md.
  // A copy is needed here so that res has its own buffer, which remains valid even after serialized_wei_desc is released.
  res.copy_(serialized_md);
  return res;
}

TORCH_LIBRARY_IMPL(mkldnn, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_reorder_convolution_transpose_weight"),
      TORCH_FN(mkldnn_reorder_conv_transpose_weight));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_reorder_linear_weight"),
      TORCH_FN(mkldnn_reorder_linear_weight));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_reorder_convolution_weight"),
      TORCH_FN(mkldnn_reorder_conv_weight));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_reorder_mkldnn_rnn_layer_weight"),
      TORCH_FN(mkldnn_reorder_mkldnn_rnn_layer_weight));
}

TORCH_LIBRARY_IMPL(mkldnn, MkldnnCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_get_mkldnn_serialized_md"),
      TORCH_FN(get_mkldnn_serialized_md ));
}

#else

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor, std::optional<ScalarType> dtype, std::optional<bool> masked_grad) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor, std::optional<ScalarType> dtype) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

Tensor mkldnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::OptionalArrayRef<int64_t> input_size) {
  TORCH_CHECK(false, "mkldnn_reorder_conv2d_weight: MKL-DNN build is disabled");
}

Tensor mkldnn_reorder_conv3d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::OptionalArrayRef<int64_t> input_size) {
  TORCH_CHECK(false, "mkldnn_reorder_conv3d_weight: MKL-DNN build is disabled");
}

#endif // AT_MKLDNN_ENABLED()

#if AT_MKL_ENABLED() && AT_MKLDNN_ENABLED()
#include <mkl.h>

static Tensor mkl_reorder_linear_weight(
    const Tensor& weight,
    const int64_t batch_size) {
  TORCH_CHECK(
      weight.scalar_type() == ScalarType::Float,
      "reorder_linear_weight: weight's dtype should be float");
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  auto M = batch_size;
  auto N = weight.size(0);
  auto K = weight.size(1);
  int64_t pack_size =
      (int64_t)(cblas_sgemm_pack_get_size(CblasBMatrix, M, N, K) / sizeof(float) + 1);
  auto packed_weight = empty_mkldnn(
      {pack_size, 1},
      weight.scalar_type(),
      weight.options().layout_opt(),
      weight.options().device_opt(),
      weight.options().pinned_memory_opt());
  ideep::tensor& mkl_weight = itensor_from_mkldnn(packed_weight);
  auto weight_ = weight.contiguous();
  const ideep::tensor orig_w = itensor_view_from_dense(weight_);
  cblas_sgemm_pack(
      CblasRowMajor,
      CblasBMatrix,
      CblasTrans,
      M,
      N,
      K,
      1.0f,
      (float*)(orig_w.get_data_handle()),
      K,
      (float*)(mkl_weight.get_data_handle()));
  return packed_weight;
}

TORCH_LIBRARY_IMPL(mkl, CPU, m) {
  m.impl(
    TORCH_SELECTIVE_NAME("mkl::_mkl_reorder_linear_weight"),
    TORCH_FN(mkl_reorder_linear_weight));
}

#endif // AT_MKL_ENABLED && AT_MKLDNN_ENABLED

}

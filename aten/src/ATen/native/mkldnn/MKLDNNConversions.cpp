#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <torch/library.h>
#include <ATen/MatrixRef.h>
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/mkldnn_reorder_conv2d_weight_native.h>
#include <ATen/ops/mkldnn_reorder_conv3d_weight_native.h>
#include <ATen/ops/to_mkldnn_native.h>
#include <ATen/ops/zeros.h>
#endif


namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor, c10::optional<ScalarType> dtype, c10::optional<bool> masked_grad) {
  TORCH_CHECK(mkldnn_tensor.scalar_type() == ScalarType::Float ||
              mkldnn_tensor.scalar_type() == ScalarType::BFloat16 ||
              mkldnn_tensor.scalar_type() == ScalarType::Byte ||
              mkldnn_tensor.scalar_type() == ScalarType::Char,
              "mkldnn_to_dense expects float, bfloat16, uint8, int8 tensor input");
  ideep::tensor& stensor = itensor_from_mkldnn(mkldnn_tensor);
  auto dims = stensor.get_dims();
  auto data_type = dtype.has_value() ? dtype.value() : mkldnn_tensor.scalar_type();
  TORCH_CHECK(data_type == ScalarType::Float ||
              data_type == ScalarType::BFloat16 ||
              data_type == ScalarType::Byte ||
              data_type == ScalarType::Char,
              "mkldnn tensor only can be converted to be a float, bfloat16, uint8, int8 cpu tensor")
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
         : (data_type == ScalarType::Byte
            ? stensor.to_public(cpu_tensor.template data_ptr<uint8_t>(),
                            ideep::tensor::data_type::u8)
            : stensor.to_public(cpu_tensor.template data_ptr<int8_t>(),
                            ideep::tensor::data_type::s8)
         )
      );
  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  return cpu_tensor.contiguous();
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(),
             "dense_to_mkldnn expects CPU tensor input");
  TORCH_CHECK(cpu_tensor.layout() == Layout::Strided,
             "dense_to_mkldnn expects strided tensor input");
  TORCH_CHECK(cpu_tensor.scalar_type() == ScalarType::Float ||
              cpu_tensor.scalar_type() == ScalarType::BFloat16 ||
              cpu_tensor.scalar_type() == ScalarType::Byte ||
              cpu_tensor.scalar_type() == ScalarType::Char,
             "dense_to_mkldnn expects float, bfloat16, uint8, int8 tensor input");
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
              data_type == ScalarType::Byte ||
              data_type == ScalarType::Char,
              "cpu tensor only can be converted to be a float, bfloat16, uint8, int8 mkldnn tensor")
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
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_reorder_conv2d_weight: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }
  const auto padding_expanded = expand_param_if_needed(padding, "padding", 2);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", 2);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", 2);
  auto w = itensor_from_mkldnn(self);

  // Legacy mkldnn conv2d jitted module may contain a 5-d weight with an extra
  // dimension when groups > 1, having dimension [g, o/g, i, h, w] instead of
  // [o, i, h, w]. Ideally we should reorder the weight back in serialization.
  // For backward compatibility, we squash the first two dims (g * o/g) back to
  // its original form.
  if (w.ndims() == 5) {
    auto wdims = w.get_dims();
    w.reshape({wdims[0] * wdims[1], wdims[2], wdims[3], wdims[4]});
  }

  ideep::dims src_dims = ideep::dims();
  bool is_channels_last = false;
  if (input_size.has_value()) {
    src_dims = input_size.value().vec();
    // if has input size, we always use channels last.
    is_channels_last = true;
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
    int64_t groups) {
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_reorder_conv3d_weight: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }
  const auto padding_expanded = expand_param_if_needed(padding, "padding", 3);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", 3);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", 3);
  auto w = itensor_from_mkldnn(self);

  auto desc =
      ideep::convolution_forward::expected_weights_desc(
          w.get_dims(),
          w.get_data_type(),
          stride_expanded,
          padding_expanded,
          padding_expanded,
          dilation_expanded,
          groups,
          ideep::algorithm::convolution_direct);
  ideep::tensor result;
  result.init(desc);
  result.feed_from(w);

  return new_with_itensor_mkldnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()), self.options().device_opt());
}

Tensor mkldnn_reorder_linear_weight(
    const Tensor& self,
    c10::optional<int64_t> batch_size_opt) {
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_reorder_linear_weight: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }
  auto out_features = self.size(0);
  auto in_features = self.size(1);
  auto w = itensor_from_mkldnn(self);
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

ideep::tensor::desc get_conv_transpose_expected_weights_desc(
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

Tensor mkldnn_reorder_conv_transpose2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::OptionalArrayRef<int64_t> input_size) {
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_reorder_conv2d_weight: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }
  const auto padding_expanded = expand_param_if_needed(padding, "padding", 2);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", 2);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", 2);
  const auto output_padding_expanded = expand_param_if_needed(output_padding, "output_padding", 2);
  ideep::tensor w = itensor_from_tensor(self);

  ideep::dims src_dims = ideep::dims();
  bool is_channels_last = false;
  if (input_size.has_value()) {
    src_dims = input_size.value().vec();
    // if has input size, we always use channels last.
    is_channels_last = true;
  }

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

std::tuple<ideep::tensor, ideep::tensor> get_lstm_packed_weights(
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

  ideep::tensor::desc packed_desc_ih, packed_desc_hh;

  std::tie(packed_desc_ih, packed_desc_hh) =
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

std::vector<Tensor> mkldnn_reorder_lstm_weight(
    TensorList weight,
    int64_t input_feature_size,
    int64_t hidden_size,
    bool has_biases,
    int64_t num_layers,
    bool bidirectional,
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

  std::vector<Tensor> result(weight.size());

  auto num_directions = bidirectional ? 2 : 1;
  int64_t weight_stride0 = has_biases ? 4 : 2;

  at::MatrixRef<at::Tensor> weights{
      weight, static_cast<size_t>(weight_stride0)};
  ideep::tensor w1_, w2_;
  at::Tensor packed_w1, packed_w2;

  for (int64_t layer = 0; layer < num_layers; layer++) {
    for (int64_t direction = 0; direction < num_directions; direction++) {
      // for layer == 0, feature_size = input_feature_size
      // otherwise, feature_size = hidden_size
      int64_t layer_feature_size = layer == 0? input_feature_size : num_directions * hidden_size;
      auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];
      TORCH_CHECK(layer_weights.size() == 2 || layer_weights.size() == 4);
      auto reverse = (direction > 0);

      std::tie(w1_, w2_) = get_lstm_packed_weights(
        layer_weights[0],
        layer_weights[1],
        has_biases ? layer_weights[2] : at::zeros(
                           layer_weights[0].sizes(),
                           layer_weights[0].options()),
        has_biases ? layer_weights[3]
                            : at::zeros(
                                  layer_weights[1].sizes(),
                                  layer_weights[1].options()),
        layer_feature_size,
        hidden_size,
        has_biases,
        num_layers,
        bidirectional,
        time_step,
        batch_size,
        reverse);

      // TODO: use is_opaque() after updating ideep in pytorch
      // Don't pack when the weight is of rnn_packed format
      // When the weight is of rnn_packed format, if the seq_lens of
      // the input changes, the format of weight also changes.
      // oneDNN does not support reorder from rnn_packed back to public format.
      // LSTM based on BRGEMM kernel (on AVX512 and newest ISAs) will use blocked
      // format for weight of LSTM, which won't change when the input seq_lens
      // changes.
      // On AVX2, queried weight will be plain format
      if (w1_.get_desc().is_rnn_packed() || w1_.get_desc().is_plain()) {
        packed_w1 = layer_weights[0];
      } else {
        packed_w1 = new_with_itensor_mkldnn(std::move(w1_), optTypeMetaToScalarType(layer_weights[0].options().dtype_opt()), layer_weights[0].options().device_opt());
      }

      if (w2_.get_desc().is_rnn_packed() || w2_.get_desc().is_plain()) {
        packed_w2 = layer_weights[1];
      } else {
        packed_w2 = new_with_itensor_mkldnn(std::move(w2_), optTypeMetaToScalarType(layer_weights[1].options().dtype_opt()), layer_weights[1].options().device_opt());
      }

      result[index * weight_stride0] = packed_w1;
      result[index * weight_stride0+1] = packed_w2;

      if (has_biases) {
        result[index * weight_stride0+2] = layer_weights[2];
        result[index * weight_stride0+3] = layer_weights[3];
      }
    }
  }

  return result;
}

TORCH_LIBRARY_IMPL(mkldnn, MkldnnCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_reorder_convolution_transpose_weight"),
      TORCH_FN(mkldnn_reorder_conv_transpose2d_weight));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_reorder_linear_weight"),
      TORCH_FN(mkldnn_reorder_linear_weight));
}

TORCH_LIBRARY_IMPL(mkldnn, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_reorder_lstm_weight"),
      TORCH_FN(mkldnn_reorder_lstm_weight));
}

#else

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor, c10::optional<ScalarType> dtype, c10::optional<bool> masked_grad) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor, c10::optional<ScalarType> dtype) {
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
    int64_t groups) {
  TORCH_CHECK(false, "mkldnn_reorder_conv3d_weight: MKL-DNN build is disabled");
}

#endif // AT_MKLDNN_ENABLED()

#if AT_MKL_ENABLED() && AT_MKLDNN_ENABLED()
#include <mkl.h>

Tensor mkl_reorder_linear_weight(
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
  ideep::tensor& orig_w = itensor_from_mkldnn(weight);
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

TORCH_LIBRARY_IMPL(mkl, MkldnnCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkl::_mkl_reorder_linear_weight"),
      TORCH_FN(mkl_reorder_linear_weight));
}

#endif // AT_MKL_ENABLED && AT_MKLDNN_ENABLED

}}

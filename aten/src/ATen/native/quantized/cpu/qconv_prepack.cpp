#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/Context.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <torch/library.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#endif

#include <c10/util/irange.h>

#ifdef USE_FBGEMM
template <int kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> PackedConvWeight<
    kSpatialDim>::
    prepack(
        at::Tensor weight,
        std::optional<at::Tensor> bias,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose) {
  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,
      "Weights are expected to have ",
      kSpatialDim + 2,
      " dimensions");
  TORCH_CHECK(
      stride.size() == kSpatialDim,
      "stride should contain ",
      kSpatialDim,
      " elements for ",
      kSpatialDim,
      "D convolution.");
  TORCH_CHECK(
      padding.size() == kSpatialDim,
      "Specify front/top/left padding only. "
      "end/bottom/right padding assumed to be equal to front/top/left");
  TORCH_CHECK(
      !transpose || output_padding.size() == kSpatialDim,
      "quantized::conv_prepack: Specify top/left output padding "
      "only. bottom/right padding assumed to be equal to top/left");
  TORCH_CHECK(
      dilation.size() == kSpatialDim,
      "dilation should contain ",
      kSpatialDim,
      " elements for ",
      kSpatialDim,
      "D convolution.");
  const int input_channels = transpose ? weight.size(0)
                                       : weight.size(1) * groups;
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  const int output_channels = transpose ? weight.size(1) * groups
                                        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
                                        : weight.size(0);
  const int kernel_d = kSpatialDim == 2 ? 1 : weight.size(2);
  const int kernel_h = weight.size(kSpatialDim);
  const int kernel_w = weight.size(kSpatialDim + 1);

  // mini-batch doesn't have any impact on how we pack weights
  // so we pass it as 1
  // Input image height/width also don't have any impact on how we pack
  // weights so we can pass any values
  const fbgemm::conv_param_t<kSpatialDim> conv_p =
      at::native::fbgemm_utils::MakeFbgemmConvParam<kSpatialDim>(
          1, // dummy batch size
          input_channels,
          output_channels,
          kSpatialDim == 2 ? std::vector<int>{28, 28} // dummy image size
                           : std::vector<int>{28, 28, 28},
          groups,
          kSpatialDim == 2 ? std::vector<int>{kernel_h, kernel_w}
                           : std::vector<int>{kernel_d, kernel_h, kernel_w},
          std::vector<int>(stride.begin(), stride.end()),
          std::vector<int>(padding.begin(), padding.end()),
          std::vector<int>(dilation.begin(), dilation.end()),
          std::vector<int>(output_padding.begin(), output_padding.end()),
          transpose);

  const auto qtype = weight.qscheme();
  std::vector<int32_t> zero_points;
  if (qtype == c10::kPerTensorAffine) {
    zero_points = {static_cast<int32_t>(weight.q_zero_point())};
  } else if (qtype == c10::kPerChannelAffine) {
    TORCH_CHECK(
        !transpose,
        "Per Channel Quantization is currently disabled for transposed conv");
    zero_points.resize(output_channels);
    for (const auto i : c10::irange(output_channels)) {
      zero_points[i] = weight.q_per_channel_zero_points()[i].item<int32_t>();
    }
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qtype));
  }

  // FBGEMM expects weights to be in channels last
  // TODO: Change this when ChannelsLast3d is ready.
  // FBGEMM needs G OC/G kDim0 ... kDimN IC/G
  // for both conv and conv transpose
  // but PyTorch lays them out as {out_c, in_c/groups, kH, kW}
  // (or for ConvTranspose {in_c, out_c/groups, kH, kW})
  const at::Tensor weight_nhwc =
      at::native::fbgemm_utils::ConvertConvWeightsToChannelLastTensor<kSpatialDim>(weight, groups, transpose);
  const int8_t* weight_data_int8 =
          reinterpret_cast<int8_t*>(weight_nhwc.data_ptr<c10::qint8>());
  std::vector<int32_t> col_offsets(output_channels);
  // compute column offsets (Similar to
  // fbgemm::col_offsets_with_zero_pt_s8acc32_ref) please note that offsets
  // include the sum of columns as well as the scalar term weight_zero_point *
  // KDim
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  const int input_channels_per_group = input_channels / groups;
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  const int output_channels_per_group = output_channels / groups;
  const int inner_size =
      kernel_d * kernel_h * kernel_w * input_channels_per_group;
  for (const auto g : c10::irange(groups)) {
    for (const auto i : c10::irange(output_channels_per_group)) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      const int c = g * output_channels_per_group + i;
      int32_t sum = 0;
      for (const auto j : c10::irange(inner_size)) {
        sum += static_cast<int32_t>(weight_data_int8[c * inner_size + j]);
      }
      if (qtype == c10::kPerTensorAffine) {
        col_offsets[c] = sum - zero_points[0] * inner_size;
      } else {
        col_offsets[c] = sum - zero_points[c] * inner_size;
      }
    }
  }

  std::vector<float> scales;
  if (qtype == c10::kPerTensorAffine) {
    scales = {static_cast<float>(weight.q_scale())};
  } else if (qtype == c10::kPerChannelAffine) {
    scales.resize(output_channels);
    for (const auto i : c10::irange(output_channels)) {
      scales[i] = weight.q_per_channel_scales()[i].item<float>();
    }
  }

  std::optional<at::Tensor> bias_contig;
  if (bias.has_value()) {
    at::Tensor bias_vec = bias.value();
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_vec.size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
    bias_contig = bias->contiguous();
  }

  auto ret_ptr = c10::make_intrusive<PackedConvWeight<kSpatialDim>>(
      PackedConvWeight<kSpatialDim>{
          std::make_unique<fbgemm::PackWeightsForConv<kSpatialDim>>(
              conv_p, weight_data_int8),
          bias_contig,
          stride,
          padding,
          output_padding,
          dilation,
          groups,
          transpose,
          col_offsets,
          kSpatialDim == 2 ? std::vector<int64_t>{kernel_h, kernel_w}
                           : std::vector<int64_t>{kernel_d, kernel_h, kernel_w},
          scales,
          zero_points,
          qtype});

  return ret_ptr;
}

template struct PackedConvWeight<2>;
template struct PackedConvWeight<3>;
#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
template <int kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> PackedConvWeightsQnnp<
    kSpatialDim>::
    prepack(
        at::Tensor weight,
        std::optional<at::Tensor> bias_in,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose) {
  TORCH_CHECK(
      kSpatialDim == 2 || kSpatialDim == 3,  // 1D is packed as 2d, hence we don't need other checks
      "QNNPACK packing only supports 2D / 3D convolution.");
  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,
      "quantized::conv_prepack (qnnpack): Weights are expected to have ",
      kSpatialDim + 2, " dimensions, found shape ", weight.sizes());
  TORCH_CHECK(
      stride.size() == kSpatialDim,
      "quantized::conv_prepack (qnnpack): ",
      kSpatialDim, "D convolution expects stride to have ",
      kSpatialDim, " elements.");
  TORCH_CHECK(
      padding.size() == kSpatialDim,
      "quantized::conv_prepack (qnnpack): Specify top/left input padding "
      "only. bottom/right padding assumed to be equal to top/left");
  TORCH_CHECK(
      !transpose || output_padding.size() == kSpatialDim,
      "quantized::conv_prepack (qnnpack): Specify top/left output padding "
      "only. bottom/right padding assumed to be equal to top/left");
  TORCH_CHECK(
      dilation.size() == kSpatialDim,
      "quantized::conv_prepack (qnnpack): ",
      kSpatialDim, "D convolution expects dilation to have ",
      kSpatialDim, " elements.");

  at::native::initQNNPACK();

  // QNNPACK expects weights to be of the format {out_c, kH, kW, in_c/groups},
  // but PyTorch lays them out as {out_c, in_c/groups, kH, kW}
  // (or for ConvTranspose {in_c, out_c/groups, kH, kW})
  const auto out_ch = transpose ? weight.size(1) * groups : weight.size(0);
  const uint32_t kernel_d = kSpatialDim == 3 ? weight.size(2) : 1;
  const uint32_t kernel_h = weight.size(kSpatialDim);
  const uint32_t kernel_w = weight.size(kSpatialDim + 1);

  at::Tensor bias_fp32;
  if (bias_in.has_value()) {
    bias_fp32 = bias_in.value();
  } else {
    bias_fp32 = at::zeros(out_ch, weight.options().dtype(at::kFloat));
  }

  TORCH_CHECK(
      !bias_fp32.defined() ||
          (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == out_ch),
      "quantized::conv2d_prepack (qnnpack): expected bias to be 1-dimensional "
      "with ",
      out_ch,
      " elements",
      ", but got bias of size ",
      bias_fp32.sizes(),
      " instead. "
      "(weight dimensions: ",
      weight.sizes(), " , transpose: ",
      (transpose ? "True)." : "False).")
  );

  TORCH_CHECK(
      !bias_fp32.defined() ||
          (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == out_ch),
      "quantized::conv3d_prepack (qnnpack): expected bias to be 1-dimensional "
      "with ",
      out_ch,
      " elements",
      ", but got bias of size ",
      bias_fp32.sizes(),
      " instead. "
      "(weight dimensions: ",
      weight.sizes(), " , transpose: ",
      (transpose ? "True)." : "False).")
  );

  auto weight_contig = weight.contiguous(
      kSpatialDim == 2 ? c10::MemoryFormat::ChannelsLast
                       : c10::MemoryFormat::ChannelsLast3d);
  const bool is_per_channel = weight_contig.qscheme() == at::kPerChannelAffine;
  auto kernel_dim = kSpatialDim == 2
      ? std::vector<int64_t>{kernel_h, kernel_w}
      : std::vector<int64_t>{kernel_d, kernel_h, kernel_w};
  auto [w_zero_points, w_scales] =
      make_zero_points_and_scales_tensor(weight_contig, transpose, groups);
  // We set the pre-packed conv weights to nullptr below as we call pre-pack
  // during the first invocation of operator run. Refer to qconv.cpp for more
  // details. TODO Update to actually call pre-pack here once bias is removed
  // from pre-packing step.
  auto ret_ptr = c10::intrusive_ptr<PackedConvWeightsQnnp<kSpatialDim>>::make(
      nullptr, /* PrePackConvWeights */
      weight_contig, /* int8_t weight */
      bias_fp32.contiguous(), /* fp32 bias */
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      transpose,
      std::nullopt, /* input_scale */
      kernel_dim,
      w_scales,
      std::move(w_zero_points),
      is_per_channel);

  return ret_ptr;
}

template
c10::intrusive_ptr<ConvPackedParamsBase<2>> PackedConvWeightsQnnp<
    2>::
    prepack(
        at::Tensor weight,
        std::optional<at::Tensor> bias_in,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose);
#endif // USE_PYTORCH_QNNPACK

#if AT_MKLDNN_ENABLED()
template <int kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> PackedConvWeightsOnednn<
    kSpatialDim>::
    prepack(
        at::Tensor weight,
        std::optional<at::Tensor> bias,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose) {
  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,
      "Weights are expected to have ", kSpatialDim + 2, " dimensions");
  TORCH_CHECK(
      stride.size() == kSpatialDim,
      "stride should contain ", kSpatialDim, " elements for ",
      kSpatialDim, "D convolution.");
  TORCH_CHECK(
      std::all_of(stride.begin(), stride.end(), [](bool s) { return s > 0; }),
      "quantized::conv_prepack: stride should be positive.");
  TORCH_CHECK(
      padding.size() == kSpatialDim,
      "Specify front/top/left padding only. "
      "end/bottom/right padding assumed to be equal to front/top/left");
  TORCH_CHECK(
      !transpose || output_padding.size() == kSpatialDim,
      "quantized::conv_prepack: Specify top/left output padding "
      "only. bottom/right padding assumed to be equal to top/left");
  TORCH_CHECK(
      dilation.size() == kSpatialDim,
      "dilation should contain ", kSpatialDim, " elements for ",
      kSpatialDim, "D convolution.");
  TORCH_CHECK(
      !transpose || std::all_of(output_padding.begin(), output_padding.end(), [](int i) { return i==0; }),
      "quantized::conv_prepack: ONEDNN only supports zero output_padding.");

  // Weight
  // Format: [OC IC//group KH KW] for conv; [IC OC//group KH KW] for deconv
  auto dims = weight.sizes().vec();
  auto strides = stride.vec();
  auto padding_l = padding.vec();
  auto padding_r = padding.vec();
  auto dilates = dilation.vec();
  auto op_attr = ideep::attr_t();
  std::vector<int32_t> wgt_zero_points;
  ideep::scale_t wgt_scales;
  const int output_channels = transpose ? weight.size(1) * groups
                                        : weight.size(0);
  const auto qtype = weight.qscheme();
  if (qtype == c10::kPerTensorAffine) {
    TORCH_CHECK(
        weight.q_zero_point()==0,
        "quantized::qconv_prepack: ONEDNN only supports symmetric quantization of weight,"
        " whose zero point must be 0.");
    wgt_zero_points = std::vector<int32_t>(1, weight.q_zero_point());
#if IDEEP_PREREQ(3, 1, 0, 1)
    wgt_scales = ideep::scale_t(1, weight.q_scale());
#elif IDEEP_PREREQ(3, 1, 0, 0)
    wgt_scales = ideep::scale_t(1, 1.0/weight.q_scale()); // Scales of ONEDNN and PyTorch are reciprocal
#else
    TORCH_CHECK(false, "Unexpected IDeep version to do qconv weight prepack.");
#endif
  } else if (qtype == c10::kPerChannelAffine) {
    TORCH_CHECK(
        !transpose,
        "Per Channel Quantization is currently disabled for transposed conv");
    wgt_zero_points.resize(output_channels);
    wgt_scales.resize(output_channels);
    for (int i = 0; i < output_channels; ++i) {
      wgt_zero_points[i] = weight.q_per_channel_zero_points()[i].item<int32_t>();
      TORCH_CHECK(
          wgt_zero_points[i]==0,
          "quantized::qconv_prepack: ONEDNN only supports symmetric quantization of weight,"
          " whose zero point must be 0.");
#if IDEEP_PREREQ(3, 1, 0, 1)
      wgt_scales[i] = weight.q_per_channel_scales()[i].item<float>();
#elif IDEEP_PREREQ(3, 1, 0, 0)
      wgt_scales[i] = 1.0f / weight.q_per_channel_scales()[i].item<float>(); // Scales of ONEDNN and PyTorch are reciprocal
#else
      TORCH_CHECK(false, "Unexpected IDeep version to do qconv weight prepack.");
#endif
    }
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qtype));
  }

  // Set runtime src zero point
  op_attr.set_zero_points_mask(DNNL_ARG_SRC, /* zero_points_mask= */0);
  at::Tensor weight_copy;
  ideep::tensor::desc w_desc;
  ideep::dims dims_iohw, dims_giohw;
  ideep::tag w_tag = ideep::tag::any;
  const bool with_groups = groups > 1;
  if (transpose) {
    // template args: <(src/dst) is_channels_last, transposed>
    w_desc = ideep::convolution_transpose_forward::expected_weights_desc<true, false>(
        dims, dnnl::memory::data_type::s8,
        strides, padding_l, padding_r, dilates, groups,
        dnnl::algorithm::deconvolution_direct, dnnl::prop_kind::forward_inference,
        ideep::dims(), op_attr);
    // convolution_transpose_forward::expected_weights_desc() gives format [i, o, ...],
    // but ONEDNN requires [o, i, ...] for computation
    dims_iohw = w_desc.get_dims();
    dims_giohw = with_groups ? ideep::utils::group_dims(dims_iohw, groups) : dims_iohw;
    std::vector<int64_t> perms(dims_giohw.size(), 0); // for permutation of weight
    std::iota(perms.begin(), perms.end(), 0);
    std::swap(perms[with_groups], perms[with_groups + 1]);
    weight_copy = weight.reshape(dims_giohw).permute(c10::IntArrayRef(perms)).clone();
  } else {
    w_desc = ideep::convolution_forward::expected_weights_desc(
        dims, dnnl::memory::data_type::s8,
        strides, padding_l, padding_r, dilates, groups,
        dnnl::algorithm::convolution_direct, dnnl::prop_kind::forward_inference,
        dnnl::memory::data_type::u8, ideep::dims(), op_attr, /*is_channels_last=*/true);
    weight_copy = weight.clone();
  }
  if (with_groups) {
    w_tag = kSpatialDim == 2 ? ideep::tag::goihw : ideep::tag::goidhw;
  } else {
    w_tag = kSpatialDim == 2 ? ideep::tag::oihw : ideep::tag::oidhw;
  }
  ideep::dims w_dims = with_groups ? ideep::utils::group_dims(w_desc.get_dims(), groups)
                                   : w_desc.get_dims();
  ideep::tensor wgt = ideep::tensor(
      ideep::tensor::desc({w_dims, dnnl::memory::data_type::s8, w_tag}, groups),
      weight_copy.data_ptr());
  wgt.set_scale(wgt_scales); // Scales are needed for feed_from().
  ideep::tensor exp_wgt;
  exp_wgt.init(w_desc);
  exp_wgt.set_scale(wgt_scales); // Also for feed_from()
  exp_wgt.feed_from(wgt, transpose); // expect wgt to be in [OC IC KH KW] format
  ideep::tensor * packed_weight_p = new ideep::tensor(std::move(exp_wgt));
  packed_weight_p->set_scale(wgt_scales);
  packed_weight_p->set_zero_point(wgt_zero_points);
  std::unique_ptr<ideep::tensor> weight_ptr(packed_weight_p);
  // Bias
  std::optional<ideep::tensor> onednn_bias{std::nullopt};
  if (bias.has_value()) {
    at::Tensor bias_vec = bias.value();
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_vec.size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
    auto bias_desc = ideep::tensor::desc(bias.value().sizes().vec(), dnnl::memory::data_type::f32);
    ideep::tensor packed_bias;
    packed_bias.init(bias_desc, bias.value().data_ptr());
    onednn_bias = std::optional<ideep::tensor>(packed_bias);
  }
  auto ret_ptr = c10::make_intrusive<PackedConvWeightsOnednn<kSpatialDim>>(
      PackedConvWeightsOnednn<kSpatialDim>{
        std::move(weight_ptr),
        onednn_bias,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        transpose
      });
  return ret_ptr;
}

template struct PackedConvWeightsOnednn<2>;
template struct PackedConvWeightsOnednn<3>;

// Return the packed weight as Mkldnn Tensor
at::Tensor _qconv_prepack_onednn(
    at::Tensor weight, // from CPU backend instead of QuantizedCPU
    at::Tensor weight_scales, // Weight zero points must be 0 for onednn
    double input_scale,
    int64_t input_zero_point,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    std::optional<torch::List<int64_t>> input_shape) {
  int kSpatialDim = weight.ndimension() - 2;
  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,
      "Weights are expected to have ", kSpatialDim + 2, " dimensions");
  TORCH_CHECK(
      stride.size() == (decltype(stride.size()))kSpatialDim,
      "stride should contain ", kSpatialDim, " elements for ",
      kSpatialDim, "D convolution.");
  TORCH_CHECK(
      padding.size() == (decltype(padding.size()))kSpatialDim,
      "Specify front/top/left padding only. "
      "end/bottom/right padding assumed to be equal to front/top/left");
  TORCH_CHECK(
      dilation.size() == (decltype(dilation.size()))kSpatialDim,
      "dilation should contain ", kSpatialDim, " elements for ",
      kSpatialDim, "D convolution.");

  bool is_1d = (1 == kSpatialDim);
  auto x_dims = input_shape.has_value()?input_shape.value().vec():ideep::dims();
  if (is_1d) {
    if (input_shape.has_value()) {
      // N, C, L -> N, C, 1, L
      x_dims.insert(x_dims.begin() + 2, 1);
    }
    if (weight.dim() == 3) {
      weight = weight.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
    }
    stride = quant_utils::MakeArgForConv1d(stride, 1);
    padding = quant_utils::MakeArgForConv1d(padding, 0);
    dilation = quant_utils::MakeArgForConv1d(dilation, 1);
    kSpatialDim += 1;
  }
  auto w_dims = weight.sizes().vec();
  auto strides = stride.vec();
  auto padding_l = padding.vec();
  auto padding_r = padding.vec();
  auto dilates = dilation.vec();
  auto op_attr = ideep::attr_t();

  ideep::scale_t weights_scales(weight_scales.numel());

  if (weight_scales.ndimension() == 0) {
    // Weight is quant per tensor, then weight_scales will be a scalar Tensor
    TORCH_CHECK(
        weight_scales.numel() == 1,
        "Weight is quant per tensor, weight scale expects 1 element but got ", weight_scales.numel(), " elements.");
#if IDEEP_PREREQ(3, 1, 0, 1)
    weights_scales[0] = weight_scales.item().toDouble();
#elif IDEEP_PREREQ(3, 1, 0, 0)
    weights_scales[0] = 1.0 / weight_scales.item().toDouble(); // Scales of ONEDNN and PyTorch are reciprocal
#else
    TORCH_CHECK(false, "Unexpected IDeep version to do qconv weight prepack.");
#endif
  } else {
    // Weight is quant per channel
    for (int i = 0; i < weight_scales.numel(); ++i) {
#if IDEEP_PREREQ(3, 1, 0, 1)
      weights_scales[i] = weight_scales[i].item().toDouble();
#elif IDEEP_PREREQ(3, 1, 0, 0)
      weights_scales[i] = 1.0 / weight_scales[i].item().toDouble();
#else
      TORCH_CHECK(false, "Unexpected IDeep version to do qconv weight prepack.");
#endif
    }
  }

  if (input_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_SRC, /* src_scales_mask= */0);
  }
  if (input_zero_point != 0) {
    op_attr.set_zero_points_mask(DNNL_ARG_SRC, /* src_zero_points_mask= */0);
  }

  at::Tensor weight_copy;
  ideep::tensor::desc w_desc;
  ideep::dims dims_iohw, dims_giohw;
  ideep::tag w_tag = ideep::tag::any;
  const bool with_groups = groups > 1;
  w_desc = ideep::convolution_forward::expected_weights_desc(
      w_dims, dnnl::memory::data_type::s8,
      strides, padding_l, padding_r, dilates, groups,
      dnnl::algorithm::convolution_direct, dnnl::prop_kind::forward_inference,
      dnnl::memory::data_type::u8, x_dims, op_attr, /*is_channels_last=*/true);

  // Note: Weight in Conv1D will unsqueeze into Conv2D in previous step
  weight_copy = weight.clone(c10::MemoryFormat::Contiguous);

  if (with_groups) {
    w_tag = kSpatialDim == 2 ? ideep::tag::goihw : ideep::tag::goidhw;
  } else {
    w_tag = kSpatialDim == 2 ? ideep::tag::oihw : ideep::tag::oidhw;
  }
  ideep::dims wei_dims = with_groups ? ideep::utils::group_dims(w_desc.get_dims(), groups)
                                  : w_desc.get_dims();
  ideep::tensor wgt = ideep::tensor(
      ideep::tensor::desc({wei_dims, dnnl::memory::data_type::s8, w_tag}, groups),
      weight_copy.data_ptr());

  wgt.set_scale(weights_scales); // Scales are needed for feed_from().

  ideep::tensor exp_wgt;
  exp_wgt.init(w_desc);
  exp_wgt.set_scale(weights_scales); // Also for feed_from()
  exp_wgt.feed_from(wgt, /*transposed*/false); // expect wgt to be in [OC IC KH KW] format

  auto packed_weight = at::native::new_with_itensor_mkldnn(
      std::move(exp_wgt),
      c10::optTypeMetaToScalarType(weight_copy.options().dtype_opt()),
      weight_copy.options().device_opt());

  return packed_weight;
}

#endif // #if AT_MKLDNN_ENABLED()

namespace at::native {
namespace {

template <int kSpatialDim = 2>
class QConvPackWeightInt8 final {
 public:
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> run_conv(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    torch::List<int64_t> output_padding;
    output_padding.reserve(kSpatialDim);
    for ([[maybe_unused]] const auto idx : c10::irange(kSpatialDim)) {
      output_padding.push_back((int64_t)0);
    }
    return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                /*transpose=*/false);
  }

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> run_deconv(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                /*transpose=*/true);
  }

 private:
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> _run(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose) {
    auto& ctx = at::globalContext();
#ifdef USE_FBGEMM
  if (ctx.qEngine() == at::QEngine::X86) {
#if AT_MKLDNN_ENABLED()
    bool use_onednn = onednn_utils::should_use_onednn_quant(
          weight, transpose, groups, output_padding);
    if (use_onednn) {
      return PackedConvWeightsOnednn<kSpatialDim>::prepack(
          weight, bias, stride, padding, output_padding, dilation, groups, transpose);
    }
#endif
      return PackedConvWeight<kSpatialDim>::prepack(
          weight, bias, stride, padding, output_padding, dilation, groups, transpose);
  } // x86
#endif // defined(USE_FBGEMM) || AT_MKLDNN_ENABLED()

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return PackedConvWeight<kSpatialDim>::prepack(
          weight, bias, stride, padding, output_padding, dilation, groups,
          transpose);
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return PackedConvWeightsQnnp<kSpatialDim>::prepack(
          weight, bias, stride, padding, output_padding, dilation, groups,
          transpose);
    }
#endif

#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      return PackedConvWeightsOnednn<kSpatialDim>::prepack(
        weight, bias, stride, padding, output_padding, dilation, groups,
            transpose);
    }
#endif

    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv2d_prepack ",
        toString(ctx.qEngine()));
  }
};



class QConv1dPackWeightInt8 final {
 public:
  static c10::intrusive_ptr<ConvPackedParamsBase<2>> run_conv(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    const torch::List<int64_t> output_padding({0});
    return _run(std::move(weight), std::move(bias), stride, padding, output_padding, dilation, groups,
                /*transpose=*/false);
  }

  static c10::intrusive_ptr<ConvPackedParamsBase<2>> run_deconv(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    return _run(std::move(weight), std::move(bias), stride, padding, output_padding, dilation, groups,
                /*transpose=*/true);
  }

 private:
  static c10::intrusive_ptr<ConvPackedParamsBase<2>> _run(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose) {
    auto& ctx = at::globalContext();
    if (weight.dim() == 3) {
      weight = weight.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
    }
    stride = quant_utils::MakeArgForConv1d(stride, 1);
    padding = quant_utils::MakeArgForConv1d(padding, 0);
    output_padding = quant_utils::MakeArgForConv1d(output_padding, 0);
    dilation = quant_utils::MakeArgForConv1d(dilation, 1);

#ifdef USE_FBGEMM
  if (ctx.qEngine() == at::QEngine::X86) {
#if AT_MKLDNN_ENABLED()
    bool use_onednn = onednn_utils::should_use_onednn_quant(
        weight, transpose, groups, output_padding);
    if (use_onednn) {
      return PackedConvWeightsOnednn<2>::prepack(
          weight, bias, stride, padding, output_padding, dilation, groups,
          transpose);
    }
#endif
    return PackedConvWeight<2>::prepack(
        std::move(weight), std::move(bias), stride, padding, output_padding, dilation, groups,
        transpose);

  } // x86
#endif

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return PackedConvWeight<2>::prepack(
          std::move(weight), std::move(bias), stride, padding, output_padding, dilation, groups,
          transpose);
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return PackedConvWeightsQnnp<2>::prepack(
          std::move(weight), std::move(bias), stride, padding, output_padding, dilation, groups,
          transpose);
    }
#endif

#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      return PackedConvWeightsOnednn<2>::prepack(
          weight, bias, stride, padding, output_padding, dilation, groups,
          transpose);
    }
#endif

    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv1d_prepack ",
        toString(ctx.qEngine()));
  }
};

class QConvPrepackOneDNN final {
 public:
  static at::Tensor run_conv(
    at::Tensor weight, // from CPU backend instead of QuantizedCPU
    at::Tensor weight_scales, // Weight zero points must be 0s for onednn
    double input_scale,
    int64_t input_zero_point,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    std::optional<torch::List<int64_t>> input_shape) {
#if AT_MKLDNN_ENABLED()
    return _qconv_prepack_onednn(
        weight, weight_scales, input_scale, input_zero_point,
        stride, padding, dilation, groups, input_shape);
#else
    TORCH_CHECK(false, "Unimplemented as onednn is not available.")
#endif
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // Conv
  // conv_prepack is deprecated, please use conv2d_prepack for 2D conv.
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_conv));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_prepack"), TORCH_FN(QConv1dPackWeightInt8::run_conv));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_conv));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_conv));
  // ConvTranspose
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_prepack"), TORCH_FN(QConv1dPackWeightInt8::run_deconv));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_deconv));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_deconv));
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  // Conv
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_conv));
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_conv));
  // ConvTranspose
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose1d_prepack"), TORCH_FN(QConv1dPackWeightInt8::run_deconv));
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_deconv));
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_deconv));
}

TORCH_LIBRARY_IMPL(onednn, CPU, m) {
  // New OP definition for Quantization in PyTorch 2.0 Export
  // Conv Prepack
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv_prepack"), TORCH_FN(QConvPrepackOneDNN::run_conv));
}

} // namespace
} // namespace at::native

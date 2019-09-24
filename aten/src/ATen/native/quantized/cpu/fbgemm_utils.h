#pragma once

#include <c10/core/QScheme.h>
#include <ATen/Tensor.h>
#include <ATen/cpp_custom_type_hack.h>
#ifdef USE_FBGEMM
#include "fbgemm/Fbgemm.h"
#include "fbgemm/QuantUtils.h"

// The struct for the packed weight matrix (PackBMatrix) and the corresponding
// column offsets used for the fully connect layer, which are both prepared in
// the prepacking step to save the computations in the inference. Note the
// column offsets include the sum of the B columns as well as the scalar term
// B_zero_point * K, whereas the row offsets created by
// PackAWithQuantRowOffset/PackAWithIm2Col/PackAWithRowOffset are only the sum
// of the A rows. The column offsets are needed for the asymmetric quantization
// (affine quantization) of input matrix.
// Note that in JIT mode we can think of a way to fuse col_offsets with bias.
struct FBGEMM_API PackedLinearWeight {
  std::unique_ptr<fbgemm::PackBMatrix<int8_t>> w;
  c10::optional<at::Tensor> bias;
  std::vector<int32_t> col_offsets;
  std::vector<float> w_scale;
  std::vector<int32_t> w_zp;
  c10::QScheme q_scheme;
};

struct FBGEMM_API PackedConvWeight {
  std::unique_ptr<fbgemm::PackWeightsForConv<2>> w;
  c10::optional<at::Tensor> bias;
  std::vector<int32_t> col_offsets;
  std::vector<int64_t> kernel;
  std::vector<float> w_scale;
  std::vector<int32_t> w_zp;
  c10::QScheme q_scheme;
};

// PackWeight: Convert the weight from uint8 to int8.
inline void convert_uint8_int8(
    int len,
    const uint8_t* src_uint8,
    int8_t* dst_int8) {
  for (int i = 0; i < len; ++i) {
    dst_int8[i] = static_cast<int8_t>(static_cast<int32_t>(src_uint8[i]) - 128);
  }
}

// UnpackWeight: Convert the weight from int8 to uint8.
inline void convert_int8_uint8(
    int len,
    const int8_t* src_int8,
    uint8_t* dst_uint8) {
  for (int i = 0; i < len; ++i) {
    dst_uint8[i] =
        static_cast<uint8_t>(static_cast<int32_t>(src_int8[i]) + 128);
  }
}

// Calculate the column offsets.
// Note this includes the sum of the columns as well as the scalar term
// B_zero_point * K, whereas the row_offsets created by
// PackAWithQuantRowOffset is only the sum of the A rows.
inline void calc_col_offsets_transpose(
    int K,
    int N,
    const int8_t* Bint8,
    int32_t* B_zero_point,
    int32_t* col_offsets,
    c10::QScheme qtype) {
  for (size_t i = 0; i < N; ++i) {
    int32_t sum = 0;
    for (size_t j = 0; j < K; ++j) {
      sum += Bint8[i * K + j];
    }
    if (qtype == c10::kPerTensorAffine) {
      col_offsets[i] = sum - B_zero_point[0] * K;
    } else {
      col_offsets[i] = sum - B_zero_point[i] * K;
    }
  }
}

static at::Tensor fbgemm_linear_prepack(
    at::Tensor weight,
    c10::optional<at::Tensor> bias) {
  TORCH_CHECK(
      weight.dim() == 2,
      "The weight tensor for quantized::linear_prepack (fbgemm) should"
      " be 2-dimensional.");

  auto N = weight.size(0);
  auto K = weight.size(1);

  // TODO: contiguous is called for further JIT optimizations.
  auto weight_contig = weight.contiguous();
  const auto qtype = weight.qscheme();
  std::vector<int32_t> weight_zero_points_int32(1, 0);
  if (qtype == c10::kPerTensorAffine) {
    weight_zero_points_int32[0] = weight.q_zero_point();
  } else if (qtype == c10::kPerChannelAffine) {
    weight_zero_points_int32.resize(N, 0);
    for (int i = 0; i < N; ++i) {
      weight_zero_points_int32[i] =
          weight.q_per_channel_zero_points()[i].item<int32_t>();
    }
  }
  std::vector<float> weight_scales_float(1, 0.0);
  if (qtype == c10::kPerTensorAffine) {
    weight_scales_float[0] = weight.q_scale();
  } else if (qtype == c10::kPerChannelAffine) {
    weight_scales_float.resize(N, 0.0);
    for (int i = 0; i < N; ++i) {
      weight_scales_float[i] = weight.q_per_channel_scales()[i].item<float>();
    }
  }

  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());

  std::vector<int32_t> col_offsets(N);
  calc_col_offsets_transpose(
      /*K=*/K,
      /*N=*/N,
      /*Bint8=*/weight_ptr_int8,
      /*B_zero_point=*/weight_zero_points_int32.data(),
      /*col_offsets=*/col_offsets.data(),
      /*qtype=*/qtype);

  c10::optional<at::Tensor> bias_contig;
  if (bias.has_value()) {
    at::Tensor bias_vec = bias.value();
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_vec.size(0) == N,
        "bias should have N elements: " + std::to_string(N));
    bias_contig = bias->contiguous();
  }
  auto ret_ptr = c10::guts::make_unique<PackedLinearWeight>(PackedLinearWeight{
      c10::guts::make_unique<fbgemm::PackBMatrix<int8_t>>(
          /*trans=*/fbgemm::matrix_op_t::Transpose,
          /*nRow=*/K,
          /*nCol=*/N,
          /*smat=*/weight_ptr_int8,
          /*ld=*/K,
          /*pmat=*/nullptr, // PackBMatrix manages ownership of pmat
          /*groups=*/1),
      bias_contig,
      col_offsets,
      weight_scales_float,
      weight_zero_points_int32,
      qtype});
  // TODO: we will need to replace this with torchscript classes at a later
  // point.
  return at::cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());
}

static std::tuple<at::Tensor, c10::optional<at::Tensor>> fbgemm_linear_unpack(
    at::Tensor packed_weight) {
  // Pull out the PackBMatrix instance from the owning tensor.
  auto& pack_ptr =
      at::cpp_custom_type_hack::cast<PackedLinearWeight>(packed_weight);
  auto packB = pack_ptr.w.get();

  int64_t N = static_cast<int64_t>(packB->numCols());
  int64_t K = static_cast<int64_t>(packB->numRows());

  at::Tensor weight_origin;
  if (pack_ptr.q_scheme == c10::kPerTensorAffine) {
    weight_origin = at::_empty_affine_quantized(
        {N, K},
        at::device(c10::kCPU).dtype(c10::kQInt8),
        pack_ptr.w_scale[0],
        pack_ptr.w_zp[0]);
  } else if (pack_ptr.q_scheme == c10::kPerChannelAffine) {
    auto scales = at::from_blob(
        pack_ptr.w_scale.data(),
        pack_ptr.w_scale.size(),
        device(c10::kCPU).dtype(c10::kFloat));
    auto zero_points = at::from_blob(
        pack_ptr.w_zp.data(), pack_ptr.w_zp.size(), device(c10::kCPU).dtype(c10::kInt));

    weight_origin = at::_empty_per_channel_affine_quantized(
        {N, K},
        scales.toType(c10::kDouble),
        zero_points.toType(c10::kLong),
        0, // The output channel axis is 0
        device(c10::kCPU).dtype(c10::kQInt8));
  }

  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

  packB->unpack(weight_ptr_int8);

  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(
      weight_origin, pack_ptr.bias);
}

static at::Tensor fbgemm_conv_prepack(
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups) {
  TORCH_CHECK(
      weight.ndimension() == 4, "Weights are expected to have 4 dimensions");
  TORCH_CHECK(stride.size() == 2, "2D convolution only");
  TORCH_CHECK(
      padding.size() == 2,
      "Specify top/left padding only. \
    bottom/right padding assumed to be equal to top/left");
  TORCH_CHECK(dilation.size() == 2, "2D convolution only");

  int output_channels = weight.size(0);
  int input_channels_per_group = weight.size(1);
  int kernel_h = weight.size(2);
  int kernel_w = weight.size(3);

  // mini-batch doesn't have any impact on how we pack weights
  // so we pass it as 1
  // Input image height/width also don't have any impact on how we pack
  // weights so we can pass any values
  fbgemm::conv_param_t<2> conv_p(
      1, // Mini-Batch
      input_channels_per_group * groups, // input channels
      output_channels,
      {28, 28}, // Image height and width
      groups,
      {kernel_h, kernel_w},
      {static_cast<int>(stride[0]), static_cast<int>(stride[1])},
      {static_cast<int>(padding[0]),
       static_cast<int>(padding[1]),
       static_cast<int>(padding[0]),
       static_cast<int>(padding[1])},
      {static_cast<int>(dilation[0]), static_cast<int>(dilation[1])});

  // FBGEMM expects weights to be in channels last
  auto weight_contig = weight.contiguous(c10::MemoryFormat::ChannelsLast);
  const auto qtype = weight.qscheme();

  std::vector<int32_t> zero_points(1, 0);
  if (qtype == c10::kPerTensorAffine) {
    zero_points[0] = weight.q_zero_point();
  } else if (qtype == c10::kPerChannelAffine) {
    int64_t axis = weight.q_per_channel_axis();
    TORCH_CHECK(
        axis == 0,
        "Only per output channel quantization is supported for the weights");
    zero_points.resize(output_channels, 0);
    for (int i = 0; i < output_channels; ++i) {
      zero_points[i] = weight.q_per_channel_zero_points()[i].item<int32_t>();
    }
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qtype));
  }
  
  const int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());

  std::vector<int32_t> col_offsets(output_channels);
  // compute column offsets (Similar to
  // fbgemm::col_offsets_with_zero_pt_s8acc32_ref) please note that offsets
  // include the sum of columns as well as the scalar term weight_zero_point *
  // KDim
  int NDim = output_channels / groups;
  int KDim_per_group = kernel_h * kernel_w * input_channels_per_group;
  for (int g = 0; g < groups; ++g) {
    for (int j = 0; j < NDim; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < KDim_per_group; ++k) {
        sum += weight_ptr_int8[(g * NDim + j) * KDim_per_group + k];
      }
      if (qtype == c10::kPerTensorAffine) {
        col_offsets[g * NDim + j] = sum - zero_points[0] * KDim_per_group;
      } else {
        col_offsets[g * NDim + j] =
            sum - zero_points[g * NDim + j] * KDim_per_group;
      }
    }
  }

  std::vector<float> scales(1, 0.0);
  if (qtype == c10::kPerTensorAffine) {
    scales[0] = weight.q_scale();
  } else if (qtype == c10::kPerChannelAffine) {
    scales.resize(output_channels, 0.0);
    for (int i = 0; i < output_channels; ++i) {
      scales[i] = weight.q_per_channel_scales()[i].item<float>();
    }
  }

  c10::optional<at::Tensor> bias_contig;
  if (bias.has_value()) {
    at::Tensor bias_vec = bias.value();
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_vec.size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
    bias_contig = bias->contiguous();
  }

  auto ret_ptr = c10::guts::make_unique<PackedConvWeight>(
      PackedConvWeight{c10::guts::make_unique<fbgemm::PackWeightsForConv<2>>(
                           conv_p, weight_ptr_int8),
                       bias_contig,
                       col_offsets,
                       {kernel_h, kernel_w},
                       scales,
                       zero_points,
                       qtype});
  // TODO: we will need to replace this with torchscript classes at a later
  // point.
  return at::cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());

}

static std::tuple<at::Tensor, c10::optional<at::Tensor>> fbgemm_conv_unpack(
    at::Tensor packed_weights) {
  // Pull out the packed weight instance from the owning tensor.
  auto& pack_ptr =
      at::cpp_custom_type_hack::cast<PackedConvWeight>(packed_weights);
  auto packed_weights_p = pack_ptr.w.get();
  // output channels
  int output_channels = packed_weights_p->outputChannels();
  int input_channels = packed_weights_p->inputChannels();
  int groups = packed_weights_p->groups();
  // R (kernel height)
  int kernel_h = pack_ptr.kernel[0];
  // S (kernel width)
  int kernel_w = pack_ptr.kernel[1];

  int C_per_G = input_channels / groups;

  // Tensor for unpacked weights
  // Unpacked format would be physical KRS(C/G) but logical KCRS (channels first)
  // because that's how FBGEMM stores the weights
  at::Tensor unpacked_weights;
  if (pack_ptr.q_scheme == c10::kPerTensorAffine) {
    unpacked_weights = at::_empty_affine_quantized(
        {output_channels, C_per_G, kernel_h, kernel_w},
        device(c10::kCPU).dtype(c10::kQInt8),
        pack_ptr.w_scale[0],
        pack_ptr.w_zp[0],
        c10::MemoryFormat::ChannelsLast);
  } else if (pack_ptr.q_scheme == c10::kPerChannelAffine) {
    auto scales = at::from_blob(
        pack_ptr.w_scale.data(),
        pack_ptr.w_scale.size(),
        device(c10::kCPU).dtype(c10::kFloat));
    auto zero_points = at::from_blob(
        pack_ptr.w_zp.data(), pack_ptr.w_zp.size(), device(c10::kCPU).dtype(c10::kInt));
    unpacked_weights = at::_empty_per_channel_affine_quantized(
        {output_channels, C_per_G, kernel_h, kernel_w},
        scales.toType(c10::kDouble),
        zero_points.toType(c10::kLong),
        0, /* The output channel axis is 0 */
        device(c10::kCPU).dtype(c10::kQInt8),
        c10::MemoryFormat::ChannelsLast);
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(pack_ptr.q_scheme));
  }

  int8_t* unpacked_weights_p =
      reinterpret_cast<int8_t*>(unpacked_weights.data_ptr<c10::qint8>());

  packed_weights_p->unpack(unpacked_weights_p);
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(
      unpacked_weights, pack_ptr.bias);
}
#endif // USE_FBGEMM

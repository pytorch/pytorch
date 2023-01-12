#include <ATen/ATen.h>

#ifdef USE_FBGEMM
#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#endif
#ifdef USE_PYTORCH_QNNPACK
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>
#endif

namespace ao {
namespace sparse {

namespace {
const int64_t serialization_version_index = 0;
const int64_t bias_index = 1;
const int64_t out_features_block_size_index = 2;
const int64_t in_features_block_size_index = 3;
const int64_t weight_scales_index = 4;
const int64_t weight_zero_point_index = 5;
const int64_t quantization_scheme_index = 6;
const int64_t row_block_indices_index = 7;
const int64_t col_block_indices_index = 8;
const int64_t weight_values_index = 9;
const int64_t num_output_channels_index = 10;
const int64_t num_input_channels_index = 11;

template <typename TENSOR_DTYPE, typename VEC_DTYPE>
std::vector<VEC_DTYPE> unwrap_vector(at::Tensor tensor) {
  std::vector<VEC_DTYPE> vec(tensor.numel());
  TENSOR_DTYPE* tensor_data_ptr = tensor.data_ptr<TENSOR_DTYPE>();
  std::copy(tensor_data_ptr, tensor_data_ptr + tensor.numel(), vec.data());
  return vec;
}

#ifdef USE_FBGEMM
/**
 * Adapted from Fbgemm BCSRMatrix::unpack, but with non-zero zero points and
 * without tiling
 * https://github.com/pytorch/FBGEMM/blob/9d7c48a65419d0350f9e9e72f31e05bfe37e85a4/src/FbgemmSparseDense.cc#L154
 */
void unpack_bcsr(
    int8_t* dst,
    ao::sparse::BCSR bcsr,
    const int64_t R,
    const int64_t C,
    const int64_t RB,
    const int64_t CB,
    const int8_t* zero_points,
    const bool qscheme_per_tensor) {
  const size_t ld = C;
  // zero out destination
  if (qscheme_per_tensor) {
    memset(dst, zero_points[0], R * C * sizeof(int8_t));
  } else {
    for (int64_t i = 0; i < R; i++) {
      memset(dst + i * C, zero_points[i], C * sizeof(int8_t));
    }
  }
  const std::vector<int8_t>& weight_values = std::get<0>(bcsr);
  const std::vector<int32_t>& row_indices = std::get<1>(bcsr);
  const std::vector<int32_t>& col_indices = std::get<2>(bcsr);
  int64_t rowBlocks = (R + RB - 1) / RB;
  for (int64_t i = 0; i < rowBlocks; ++i) {
    // For the current tile, rowBPtr starts from currentTileIdx
    for (int64_t r = row_indices[i]; r < row_indices[i + 1]; ++r) {
      int64_t curColIdx = col_indices[r];
      for (int64_t ib = 0; ib < RB; ++ib) {
        for (int64_t jb = 0; jb < CB; ++jb) {
          // Are we within bounds of destination matrix?
          if ((i * RB + ib) < R && (curColIdx * CB + jb) < C) {
            dst[(i * RB + ib) * ld + curColIdx * CB + jb] =
                weight_values[r * RB * CB + ib * CB + jb];
          }
        }
      }
    }
  }
}
#endif // USE_FBGEMM
} // namespace

#ifdef USE_FBGEMM

c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeight::deserialize(
    const BCSRSerializationType& serialized) {
  const int64_t out_features_block_size =
      std::get<out_features_block_size_index>(serialized);
  const int64_t in_features_block_size =
      std::get<in_features_block_size_index>(serialized);
  const c10::QScheme q_scheme = std::get<quantization_scheme_index>(serialized)
      ? c10::kPerTensorAffine
      : c10::kPerChannelAffine;
  const int64_t output_channels =
      std::get<num_output_channels_index>(serialized);
  const int64_t input_channels = std::get<num_input_channels_index>(serialized);
  // Unpack the untiled bcsr, then pack it in tiled form
  at::Tensor weight_origin;
  const at::Tensor weight_zero_points =
      std::get<weight_zero_point_index>(serialized);
  if (q_scheme == c10::kPerTensorAffine) {
    weight_origin = at::_empty_affine_quantized(
        {output_channels, input_channels},
        at::device(c10::kCPU).dtype(c10::kQInt8),
        std::get<weight_scales_index>(serialized).data_ptr<float>()[0],
        weight_zero_points.data_ptr<int8_t>()[0]);
  } else if (q_scheme == c10::kPerChannelAffine) {
    weight_origin = at::_empty_per_channel_affine_quantized(
        {output_channels, input_channels},
        std::get<weight_scales_index>(serialized),
        weight_zero_points,
        0, // The output channel axis is 0
        device(c10::kCPU).dtype(c10::kQInt8));
  }

  const at::Tensor loaded_weight_values =
      std::get<weight_values_index>(serialized);
  const uint8_t* loaded_weight_values_ptr =
      loaded_weight_values.data_ptr<uint8_t>();
  const int64_t loaded_weight_values_size = loaded_weight_values.numel();
  // Subtract 128 because we serialize as +128, which s best for
  // minimizing memory footprint for QNNPack
  std::vector<int8_t> weight_values(loaded_weight_values_size);
  std::transform(
      loaded_weight_values_ptr,
      loaded_weight_values_ptr + loaded_weight_values_size,
      weight_values.begin(),
      [](uint8_t v) {
        return static_cast<int8_t>(static_cast<int16_t>(v) - 128);
      });

  const at::Tensor row_block_indices =
      std::get<row_block_indices_index>(serialized);
  const at::Tensor col_block_indices =
      std::get<col_block_indices_index>(serialized);
  // Unpack as non backend specific untiled BCSR then pack as Fbgemm tiled BCSR
  // because untiled Fbgemm BCSR currently doesn't exist
  unpack_bcsr(
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>()),
      AT_DISPATCH_INTEGRAL_TYPES(
          row_block_indices.scalar_type(),
          "packed_linear_weight_fbgemm_setup_bcsr",
          [&] {
            return ao::sparse::BCSR(
                std::move(weight_values),
                unwrap_vector<scalar_t, int32_t>(
                    std::get<row_block_indices_index>(serialized)),
                unwrap_vector<scalar_t, int32_t>(
                    std::get<col_block_indices_index>(serialized)));
          }),
      output_channels,
      input_channels,
      out_features_block_size,
      in_features_block_size,
      weight_zero_points.data_ptr<int8_t>(),
      q_scheme == c10::kPerTensorAffine);

  return PackedLinearWeight::prepack(
      weight_origin,
      std::get<bias_index>(serialized),
      out_features_block_size,
      in_features_block_size);
}

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK

c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightQnnp::deserialize(
    const BCSRSerializationType& serialized) {
  return c10::make_intrusive<PackedLinearWeightQnnp>(serialized);
}

template <typename INDICES_DTYPE>
struct UnsignedIndicesTypeTrait {
  static_assert(
      sizeof(INDICES_DTYPE) == 0,
      "Invalid dtype for UnsignedIndicesTypeTrait");
};

template <>
struct UnsignedIndicesTypeTrait<int32_t> {
  using t = uint32_t;
};

template <>
struct UnsignedIndicesTypeTrait<int16_t> {
  using t = uint16_t;
};

template <>
struct UnsignedIndicesTypeTrait<int8_t> {
  using t = uint8_t;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
PackedLinearWeightQnnp::PackedLinearWeightQnnp(
    const BCSRSerializationType& serialized)
    : LinearPackedParamsBase(
          std::get<out_features_block_size_index>(serialized),
          std::get<in_features_block_size_index>(serialized)),
      orig_bias_(std::get<bias_index>(serialized)),
      q_scheme_(
          std::get<quantization_scheme_index>(serialized)
              ? c10::kPerTensorAffine
              : c10::kPerChannelAffine),
      output_channels_(std::get<num_output_channels_index>(serialized)),
      input_channels_(std::get<num_input_channels_index>(serialized)) {
  const int64_t serialization_version =
      std::get<serialization_version_index>(serialized);
  TORCH_CHECK(
      serialization_version <= SPARSE_LINEAR_PACKED_PARAM_SERIALIZATION_VERSION,
      "Attempted to deserialize sparse qlinear packed params with an ",
      "incompatible serialization version (",
      serialization_version,
      " > ",
      SPARSE_LINEAR_PACKED_PARAM_SERIALIZATION_VERSION,
      ")");

  if (orig_bias_.has_value()) {
    bias_ = orig_bias_.value();

    TORCH_CHECK(
        (bias_.ndimension() == 1 && bias_.size(0) == output_channels_),
        "ao::sparse::qlinear_deserialize (qnnpack): Given weight of size ",
        "{",
        output_channels_,
        ", ",
        input_channels_,
        "}",
        ", expected bias to be 1-dimensional with ",
        output_channels_,
        " elements",
        ", but got bias of size ",
        bias_.sizes(),
        " instead");
  } else {
    bias_ = at::zeros(output_channels_, at::device(at::kCPU).dtype(at::kFloat));
  }

  // Pad amount (8) comes from make_zero_points_and_scales_tensor
  // https://github.com/pytorch/pytorch/blob/f8c1acea1e78573c04cd18893c4abff9eea64b03/aten/src/ATen/native/quantized/cpu/qnnpack_utils.h#L468
  const int64_t output_channels_padded = output_channels_ + 8;

  w_scales_ = at::empty(
      {output_channels_padded}, at::device(at::kCPU).dtype(at::kFloat));
  float* w_scales_data_ptr = w_scales_.data_ptr<float>();
  std::fill_n(
      w_scales_data_ptr + output_channels_,
      output_channels_padded - output_channels_,
      1); // Pad with 1

  w_zero_points_ =
      std::vector<uint8_t>(output_channels_padded, 0); // Pad with 0;

  const float* w_scales_orig_data_ptr =
      std::get<weight_scales_index>(serialized).data_ptr<float>();
  const int8_t* w_zp_orig_data_ptr =
      std::get<weight_zero_point_index>(serialized).data_ptr<int8_t>();

  const std::function<uint8_t(int8_t)> add_128 = [](int8_t v) {
    return static_cast<uint8_t>(static_cast<int16_t>(v) + 128);
  };

  if (q_scheme_ == at::kPerTensorAffine) {
    std::fill_n(w_scales_data_ptr, output_channels_, w_scales_orig_data_ptr[0]);
    std::fill_n(
        w_zero_points_.begin(), output_channels_, w_zp_orig_data_ptr[0] + 128);
  } else if (q_scheme_ == at::kPerChannelAffine) {
    std::copy(
        w_scales_orig_data_ptr,
        w_scales_orig_data_ptr + output_channels_,
        w_scales_data_ptr);
    std::transform(
        w_zp_orig_data_ptr,
        w_zp_orig_data_ptr + output_channels_,
        w_zero_points_.begin(),
        add_128);
  } else {
    TORCH_CHECK(false, "Unsupported quantization scheme.");
  }

  deserialized_bcsr_row_block_indices_ =
      std::get<row_block_indices_index>(serialized);
  deserialized_bcsr_col_block_indices_ =
      std::get<col_block_indices_index>(serialized);
  deserialized_bcsr_weight_values_ = std::get<weight_values_index>(serialized);

#define AT_DISPATCH_CASE_BCSR_INDICES_TYPES(...)      \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)

#define AT_DISPATCH_BCSR_INDICES_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                   \
      TYPE, NAME, AT_DISPATCH_CASE_BCSR_INDICES_TYPES(__VA_ARGS__))

  bcsr_matrix_ = AT_DISPATCH_BCSR_INDICES_TYPES(
      deserialized_bcsr_row_block_indices_.scalar_type(),
      "packed_linear_weight_qnnp_setup_bcsr",
      [&] {
        using unsigned_t = UnsignedIndicesTypeTrait<scalar_t>::t;
        return qnnpack::generateBlockCSRMatrix<unsigned_t>(
            reinterpret_cast<unsigned_t*>(
                deserialized_bcsr_col_block_indices_.data_ptr<scalar_t>()),
            reinterpret_cast<unsigned_t*>(
                deserialized_bcsr_row_block_indices_.data_ptr<scalar_t>()),
            deserialized_bcsr_weight_values_.data_ptr<uint8_t>(),
            deserialized_bcsr_col_block_indices_.numel(),
            deserialized_bcsr_row_block_indices_.numel(),
            deserialized_bcsr_weight_values_.numel(),
            out_features_block_size_,
            in_features_block_size_);
      });

#undef AT_DISPATCH_CASE_BCSR_INDICES_TYPES
#undef AT_DISPATCH_BCSR_INDICES_TYPES
}
#endif // USE_PYTORCH_QNNPACK

} // namespace sparse
} // namespace ao

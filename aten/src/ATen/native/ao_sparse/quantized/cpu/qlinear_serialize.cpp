#include <ATen/ATen.h>

#ifdef USE_FBGEMM
#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#endif
#ifdef USE_PYTORCH_QNNPACK
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>

#include <utility>
#endif

namespace ao {
namespace sparse {

namespace {
/**
  - Wrap a vector in a Tensor, copying data into its own data pointer.
  - The type of vec is T& (not vector<T>&) so this works with any vector-like
    datastructure which has .data() and .size()
 */
template <typename UNDERLYING_DTYPE, typename T>
at::Tensor wrap_vector(T& vec, c10::ScalarType dtype) {
  at::Tensor t = at::empty(
      {static_cast<long>(vec.size())}, at::device(c10::kCPU).dtype(dtype));
  std::copy(
      vec.data(), vec.data() + vec.size(), t.data_ptr<UNDERLYING_DTYPE>());
  return t;
}

#ifdef USE_FBGEMM
/**
 * Adapted from Fbgemm BCSRMatrix::pack, but with zero points, without tiling,
 * and without determining row_offsets
 * https://github.com/pytorch/FBGEMM/blob/9d7c48a65419d0350f9e9e72f31e05bfe37e85a4/src/FbgemmSparseDense.cc#L84
 */
ao::sparse::BCSR pack_bcsr(
    const int8_t* src,
    const int64_t R,
    const int64_t C,
    const int64_t RB,
    const int64_t CB,
    const int8_t* zero_points,
    const bool qscheme_per_tensor) {
  const size_t ld = C;
  std::vector<int32_t> rowBPtr;
  std::vector<int32_t> colBIdx;
  std::vector<int8_t> values;
  rowBPtr.push_back(0);
  int64_t nnzb = 0;
  int64_t rowBlocks = (R + RB - 1) / RB;
  for (int64_t i = 0; i < rowBlocks; ++i) {
    int64_t curCols = C;
    int64_t curColBlocks = (curCols + CB - 1) / CB;
    for (int64_t j = 0; j < curColBlocks; ++j) {
      // is the whole block zero?
      bool isCurrentBlockNonZero = false;
      for (int64_t ib = 0; ib < RB; ++ib) {
        // break if already found a non-zero element or
        // out of bounds
        if (isCurrentBlockNonZero || (i * RB + ib) >= R) {
          break;
        }
        const int64_t curr_row = i * RB + ib;
        const int8_t curr_row_zero_point =
            qscheme_per_tensor ? zero_points[0] : zero_points[curr_row];
        for (int64_t jb = 0; jb < CB; ++jb) {
          // within bound?
          if ((j * CB + jb) >= C) {
            continue;
          } else {
            if (src[curr_row * ld + j * CB + jb] != curr_row_zero_point) {
              isCurrentBlockNonZero = true;
              break;
            }
          }
        }
      }
      if (isCurrentBlockNonZero) {
        for (int64_t ib = 0; ib < RB; ++ib) {
          for (int64_t jb = 0; jb < CB; ++jb) {
            if ((i * RB + ib) >= R || (j * CB + jb) >= C) {
              // zero fill
              values.push_back(0);
            } else {
              int8_t val = src[(i * RB + ib) * ld + j * CB + jb];
              values.push_back(val);
            }
          }
        }
        colBIdx.push_back(static_cast<int32_t>(j));
        nnzb++;
      }
    }
    rowBPtr.push_back(static_cast<int32_t>(nnzb));
  }
  return ao::sparse::BCSR(
      std::move(values), std::move(rowBPtr), std::move(colBIdx));
}
#endif // USE_FBGEMM
} // namespace

#ifdef USE_FBGEMM

BCSRSerializationType PackedLinearWeight::serialize() {
  // Get weights, row indices, and col indices in untiled form;
  // unpack the tiled bcsr then pack it in untiled form
  std::vector<int8_t> dense_weight_values = std::vector<int8_t>(w->R * w->C);
  w->unpack(dense_weight_values.data());

  const bool qscheme_per_tensor = (q_scheme == c10::kPerTensorAffine);
  at::Tensor zero_points = wrap_vector<int8_t>(w_zp, c10::kChar);

  ao::sparse::BCSR untiled_bcsr = pack_bcsr(
      dense_weight_values.data(),
      w->R,
      w->C,
      w->RB,
      w->CB,
      zero_points.data_ptr<int8_t>(),
      qscheme_per_tensor);

  std::vector<int8_t>& packed_weight_values = std::get<0>(untiled_bcsr);
  // Add 128 to each weight value. This serialization format is best for
  // minimizing memory footprint for QNNPack

  at::Tensor weight_values = at::empty(
      {static_cast<long>(packed_weight_values.size())},
      at::device(c10::kCPU).dtype(c10::kByte));
  std::transform(
      packed_weight_values.begin(),
      packed_weight_values.end(),
      weight_values.data_ptr<uint8_t>(),
      [](int8_t v) {
        return static_cast<uint8_t>(static_cast<int16_t>(v) + 128);
      });

  return BCSRSerializationType(
      SPARSE_LINEAR_PACKED_PARAM_SERIALIZATION_VERSION,
      bias_,
      out_features_block_size_,
      in_features_block_size_,
      wrap_vector<float>(w_scale, c10::kFloat),
      // Narrowing from int32_t to int8_t; this is okay because qint8 zero
      // points are restricted to fit in bounds of int_8
      std::move(zero_points),
      qscheme_per_tensor,
      wrap_vector<int>(
          std::get<1>(untiled_bcsr), c10::kInt), // Row block indices
      wrap_vector<int>(
          std::get<2>(untiled_bcsr), c10::kInt), // Col block indices
      std::move(weight_values),
      w->R,
      w->C);
}

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK

BCSRSerializationType PackedLinearWeightQnnp::serialize() {
  at::Tensor w_scales_compact;
  at::Tensor w_zero_points_compact;
  const float* w_scales_data_ptr = w_scales_.data_ptr<float>();
  std::function<int8_t(uint8_t)> subtract_128 = [](uint8_t v) {
    return static_cast<int8_t>(static_cast<int16_t>(v) - 128);
  };

  if (q_scheme_ == at::kPerTensorAffine) {
    w_scales_compact = at::empty({1}, at::device(c10::kCPU).dtype(c10::kFloat));
    w_zero_points_compact =
        at::empty({1}, at::device(c10::kCPU).dtype(c10::kChar));

    w_scales_compact.data_ptr<float>()[0] = w_scales_data_ptr[0];
    w_zero_points_compact.data_ptr<int8_t>()[0] =
        static_cast<int8_t>(static_cast<int16_t>(w_zero_points_[0]) - 128);
  } else if (q_scheme_ == at::kPerChannelAffine) {
    w_scales_compact =
        at::empty({output_channels_}, at::device(c10::kCPU).dtype(c10::kFloat));
    w_zero_points_compact =
        at::empty({output_channels_}, at::device(c10::kCPU).dtype(c10::kChar));

    std::copy(
        w_scales_data_ptr,
        w_scales_data_ptr +
            output_channels_, // Don't go to the end because of padding
        w_scales_compact.data_ptr<float>());

    // Subtract 128 from each zero point, to reverse addition done during
    // prepacking
    std::transform(
        w_zero_points_.begin(),
        w_zero_points_.begin() +
            output_channels_, // Don't go to the end because of padding
        w_zero_points_compact.data_ptr<int8_t>(),
        std::move(subtract_128));
  } else {
    TORCH_CHECK(false, "Unsupported quantization scheme.");
  }

  at::Tensor wrapped_row_values;
  at::Tensor wrapped_col_indices;

  const uint32_t max_index = bcsr_matrix_->max_index();

  if (max_index <= std::numeric_limits<uint8_t>::max()) {
    // Cast from uint8_t range to int8_t
    wrapped_row_values = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int8_t>(typed_bcsr->row_values, c10::kChar); });
    wrapped_col_indices = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int8_t>(typed_bcsr->col_indices, c10::kChar); });
  } else if (max_index <= std::numeric_limits<uint16_t>::max()) {
    // Cast from uint16_t range to int16_t
    wrapped_row_values = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int16_t>(typed_bcsr->row_values, c10::kShort); });
    wrapped_col_indices = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int16_t>(typed_bcsr->col_indices, c10::kShort); });
  } else {
    // Cast from uint32_t range to int32_t
    wrapped_row_values = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int>(typed_bcsr->row_values, c10::kInt); });
    wrapped_col_indices = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int>(typed_bcsr->col_indices, c10::kInt); });
  }

  return BCSRSerializationType(
      SPARSE_LINEAR_PACKED_PARAM_SERIALIZATION_VERSION,
      orig_bias_,
      out_features_block_size_,
      in_features_block_size_,
      std::move(w_scales_compact),
      std::move(w_zero_points_compact),
      (q_scheme_ == c10::kPerTensorAffine),
      wrapped_row_values,
      wrapped_col_indices,
      wrap_vector<uint8_t>(bcsr_matrix_->values, c10::kByte),
      output_channels_,
      input_channels_);
}

#endif // USE_PYTORCH_QNNPACK

} // namespace sparse
} // namespace ao

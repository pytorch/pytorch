#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>

#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/custom_class.h>
#include <ATen/native/quantized/cpu/EmbeddingPackedParams.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/choose_qparams_optimized.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/resize_native.h>
#endif

#include <c10/util/irange.h>

#include <utility>

int register_embedding_params();

/*
 * Prepack function for embedding_bag weights.
 * This function expects a per-row quantized weight tensor
 * with a floating point scale and zero_point value.
 * zero point is set to be (-Xmin/scale)
 * To prepack the weights we store the scale and bias (where bias is Xmin)
 * for each row along with the quantized weights.
 */
c10::intrusive_ptr<EmbeddingPackedParamsBase> PackedEmbeddingBagWeight::prepack(
    at::Tensor qweight) {
  static constexpr int64_t version = 1;
  TORCH_CHECK(
      qweight.dim() == 2,
      "quantized::embedding_bag_prepack weight tensor rank should be 2");
  TORCH_CHECK(
      qweight.scalar_type() == c10::kQUInt8 ||
          qweight.scalar_type() == c10::kQUInt4x2,
      "qembedding_bag_prepack currently only supports quint8 and quint4x2 weights");

  at::Tensor weight_contig =
      qweight.contiguous(qweight.suggest_memory_format());

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int bit_width, scale_bias_bytes;
  uint8_t* weight_data = static_cast<uint8_t*>(weight_contig.data_ptr());
  if (qweight.scalar_type() == c10::kQUInt8) {
    bit_width = 8;
    scale_bias_bytes =
        sizeof(float) * 2; // extra 8 bytes to store FP scale and bias per row.
  } else {
    bit_width = 4;
    scale_bias_bytes = sizeof(at::Half) *
        2; // extra 4 bytes to store at::Half scale and bias per row.
  }
  const auto num_elem_per_byte = 8 / bit_width;

  int64_t embedding_rows = qweight.size(0);
  int64_t embedding_cols = qweight.size(1);
  const auto qtype = qweight.qscheme();
  TORCH_CHECK(
      qtype == c10::kPerChannelAffineFloatQParams,
      "Expect embedding_bag weights to be quantized using kPerChannelAffineFloatQParams");
  std::vector<float> weight_bias(embedding_rows);

  at::Tensor channel_scales = qweight.q_per_channel_scales();
  at::Tensor channel_zero_points = qweight.q_per_channel_zero_points();
  std::vector<float> weight_scales(
      channel_scales.data_ptr<float>(),
      channel_scales.data_ptr<float>() + embedding_rows);
  std::vector<float> weight_zero_points(
      channel_zero_points.data_ptr<float>(),
      channel_zero_points.data_ptr<float>() + embedding_rows);

  for (const auto i : c10::irange(embedding_rows)) {
    weight_bias[i] = weight_zero_points[i] * weight_scales[i] * -1;
  }

  std::vector<int64_t> output_shape = {
      embedding_rows,
      static_cast<std::int64_t>(
          (embedding_cols + num_elem_per_byte - 1) / num_elem_per_byte +
          scale_bias_bytes)}; // extra bytes to store scale and bias per row.
  size_t output_columns = output_shape[1];

  // Allocate output packed weights.
  at::Tensor output = at::empty(
      output_shape,
      weight_contig.options().dtype(at::kByte),
      weight_contig.suggest_memory_format());
  auto* output_data = output.data_ptr<uint8_t>();

  if (bit_width == 8) {
    at::parallel_for(
        0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
          for (const auto row : c10::irange(start_idx, end_idx)) {
            const uint8_t* input_row = weight_data + row * embedding_cols;
            std::uint8_t* output_row = output_data + row * output_columns;
            auto output_row_scale_bias = output_row + embedding_cols;
            // don't use float* to avoid unaligned address access
            std::memcpy(
                output_row_scale_bias, &(weight_scales[row]), sizeof(float));
            std::memcpy(
                output_row_scale_bias + sizeof(float),
                &(weight_bias[row]),
                sizeof(float));
            for (const auto col : c10::irange(embedding_cols)) {
              output_row[col] = input_row[col];
            }
          }
        });
  } else {
    // Re-calculate the number of embedding_cols, to account for values packed
    // in a byte.
    embedding_cols =
        (embedding_cols + num_elem_per_byte - 1) / num_elem_per_byte;
    at::parallel_for(
        0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
          for (const auto row : c10::irange(start_idx, end_idx)) {
            const uint8_t* input_row = weight_data + row * embedding_cols;
            std::uint8_t* output_row = output_data + row * output_columns;
            auto output_row_scale_bias = output_row + embedding_cols;
            at::Half weight_scale = weight_scales[row];
            at::Half weight_bias_half = weight_bias[row];
            // don't use at::Half* to avoid unaligned address access
            std::memcpy(output_row_scale_bias, &weight_scale, sizeof(at::Half));
            std::memcpy(
                output_row_scale_bias + sizeof(at::Half),
                &weight_bias_half,
                sizeof(at::Half));

            for (const auto col : c10::irange(embedding_cols)) {
              // The weight values have already been packed, so here we just
              // store it in the output tensor.
              output_row[col] = input_row[col];
            }
          }
        });
  }

  auto packed_ptr = c10::make_intrusive<PackedEmbeddingBagWeight>(
      output,
      std::move(weight_scales),
      std::move(weight_zero_points),
      bit_width,
      qtype,
      version);

  return packed_ptr;
}

namespace at {
namespace native {

// Note - This is a temporary pack function for embedding bag which quantizes
// and packs the float weight tensor. In the next step it will be replaced by a
// quantize and pack function once we support FP scale and FP zero_point
//
// Python example examining a packed 8bit zero_point and scale:
//
// >> x = torch.from_numpy(np.array([[[10, 20], [30, 40]],[[50, 60], [70, 80]]],
// dtype=np.float32))
// >> x_packed = torch.ops.quantized.embedding_bag_byte_prepack(x)
//
// # Pull out and examine packed scales, zero_points and values
// >> zero_points = x_packed[:,:,-4:].numpy()
// >> scales = x_packed[:,:,-8:-4].numpy()
// >> values = x_packed[:,:,:-8].numpy()
//
// >> zero_points
// array([[[  0,   0,  32,  65],
//        [  0,   0, 240,  65]],
//
//       [[  0,   0,  72,  66],
//        [  0,   0, 140,  66]]], dtype=uint8)
//
// >> scales
// array([[[161, 160,  32,  61],
//        [161, 160,  32,  61]],
//
//       [[161, 160,  32,  61],
//        [161, 160,  32,  61]]], dtype=uint8)
// >> values
// array([[[  0, 255],
//        [  0, 255]],
//
//       [[  0, 255],
//        [  0, 255]]], dtype=uint8)
//
// # Convert 4 byte packed scales and zero_points to float
// # and apply against values in order to recover unquantized values.
// def bytes2float(arr):
//    packed_hex = bytearray(arr)
//    return struct.unpack('f', packed_hex)
//
// >> float_zero_points = np.apply_along_axis(bytes2float, 2, zero_points)
// >> float_zero_points
// array([[[10.],
//         [30.]],
//
//        [[50.],
//         [70.]]])
// >> float_scales = np.apply_along_axis(bytes2float, 2, scales)
// >> float_scales
// array([[[0.03921569],
//        [0.03921569]],
//
//       [[0.03921569],
//        [0.03921569]]])
// >> values *  float_scales + float_zero_points
// array([[[10.        , 20.00000035],
//         [30.        , 40.00000035]],
//
//        [[50.        , 60.00000035],
//         [70.        , 80.00000035]]])
Tensor& qembeddingbag_byte_prepack_out(Tensor& output, const Tensor& weight) {
  // The "last" dimension of an N-Dimensioned batch of embedding bags is
  // quantization channel. E.g. for a 2D embedding bag, this has
  // [ row, col ] dimensions, for batched of embedding bags, dimensions might be
  // [ batch, row, col ].
  //
  // Python Batched Embedding Example:
  // weights = torch.from_numpy((np.random.random_sample((
  //          2, 10, 3)).squeeze() + 1).astype(np.float32))
  // assert(weights.size() == torch.Size([2, 10, 3]))
  // # NOTE: 8 bytes (columns) are added due to fp32 zero_point and scales
  // packed_weights = torch.ops.quantized.embedding_bag_byte_prepack(weights)
  // assert(packed_weights.size() == torch.Size([2, 10, 11]))

  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float ||
          weight.scalar_type() == at::ScalarType::Half,
      "'embedding_bag_byte_prepack' only support float32 or float16.");

  const auto weight_sizes = weight.sizes();
  const auto cols_dim = weight_sizes.size() - 1;
  const int64_t embedding_rows = c10::size_to_dim_(cols_dim, weight_sizes);
  const int32_t embedding_cols = weight_sizes[cols_dim];
  // Add 8 bytes per column to store FP32 scale and zero_point per row.
  const int32_t output_columns = embedding_cols + 2 * sizeof(float);
  const auto weight_contig =
      weight.expect_contiguous(weight.suggest_memory_format());

  // Adjust output dimensions to account for FP32 scale and zero_points.
  std::vector<int64_t> output_shape = weight_sizes.vec();
  output_shape[cols_dim] = output_columns;
  at::native::resize_(output, output_shape, c10::nullopt);
  auto* output_data = output.data_ptr<uint8_t>();

#ifdef USE_FBGEMM
  if (weight_contig->scalar_type() == at::ScalarType::Half) {
    const auto weight_data =
        static_cast<fbgemm::float16*>(weight_contig->data_ptr());
    at::parallel_for(
        0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
          fbgemm::FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<
              fbgemm::float16>(
              weight_data + start_idx * embedding_cols,
              end_idx - start_idx,
              embedding_cols,
              output_data + start_idx * output_columns);
        });
  } else {
    const auto weight_data = weight_contig->data_ptr<float>();
    at::parallel_for(
        0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
          fbgemm::FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<float>(
              weight_data + start_idx * embedding_cols,
              end_idx - start_idx,
              embedding_cols,
              output_data + start_idx * output_columns);
        });
  }

#else
  const Tensor& float_weight =
      weight_contig->scalar_type() == at::ScalarType::Half
      ? weight_contig->to(at::ScalarType::Float)
      : *weight_contig;
  const auto weight_data = float_weight.data_ptr<float>();
  constexpr float kEpsilon = 1e-8f;
  for (auto row : c10::irange(embedding_rows)) {
    const float* input_row = weight_data + row * embedding_cols;
    std::uint8_t* output_row = output_data + row * output_columns;
    float* output_row_scale_zp =
        reinterpret_cast<float*>(output_row + embedding_cols);

    float minimum_element =
        *std::min_element(input_row, input_row + embedding_cols);
    float maximum_element =
        *std::max_element(input_row, input_row + embedding_cols);
    float range = maximum_element - minimum_element;

    output_row_scale_zp[0] = range / 255.0f;
    output_row_scale_zp[1] = minimum_element;
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    for (auto col : c10::irange(embedding_cols)) {
      output_row[col] =
          lrintf((input_row[col] - minimum_element) * inverse_scale);
    } // embedding_cols
  } // embedding_rows
#endif // USE_FBGEMM

  return output;
}

Tensor qembeddingbag_byte_prepack(const Tensor& weight) {
  const auto weight_contig =
      weight.expect_contiguous(weight.suggest_memory_format());
  Tensor output = at::detail::empty_cpu(
      {0},
      at::kByte,
      weight_contig->layout(),
      weight_contig->device(),
      c10::nullopt,
      c10::nullopt);
  qembeddingbag_byte_prepack_out(output, weight);
  return output;
}

Tensor qembeddingbag_byte_prepack_meta(const Tensor& weight) {
  const auto weight_contig =
      weight.expect_contiguous(weight.suggest_memory_format());
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float ||
          weight.scalar_type() == at::ScalarType::Half,
      "'embedding_bag_byte_prepack' only support float32 or float16.");
  const auto weight_sizes = weight.sizes();
  const auto cols_dim = weight_sizes.size() - 1;
  const int32_t embedding_cols = weight_sizes[cols_dim];
  // Add 8 bytes per column to store FP32 scale and zero_point per row.
  const int32_t output_columns = embedding_cols + 2 * sizeof(float);

  // Adjust output dimensions to account for FP32 scale and zero_points.
  std::vector<int64_t> output_shape = weight_sizes.vec();
  output_shape[cols_dim] = output_columns;
  at::SymDimVector output_shape_vec(output_shape);

  return at::empty_symint(output_shape_vec, weight.options().dtype(weight.scalar_type()), weight.suggest_memory_format());
}

namespace {

// TODO: Extend support to N-D batched embeddings, similar to
// qembeddingbag_byte_prepack
Tensor _qembeddingbag_nbit_prepack_helper(
    const Tensor& weight,
    int bit_width,
    const bool optimized_qparams,
    const int64_t nbins,
    const double ratio) {
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float ||
          weight.scalar_type() == at::ScalarType::Half,
      "'qembeddingbag_nbit_prepack' only support float32 or float16.");

  int64_t embedding_rows = weight.size(0);
  int64_t embedding_cols = weight.size(1);

  Tensor weight_contig = weight.contiguous(weight.suggest_memory_format());

  TORCH_CHECK(
      bit_width == 4 || bit_width == 2,
      "bit_width must be either 2 or 4 to use 'qembeddingbag_nbit_prepack'."
      "For 8bit, consider using 'embedding_bag_byte_prepack'.");

  int NUM_ELEM_PER_BYTE = 8 / bit_width;
  TORCH_CHECK(
      weight_contig.size(weight.dim() - 1) % NUM_ELEM_PER_BYTE == 0,
      "qembeddingbag_" + c10::to_string(bit_width) +
          "bit_prepack only works for the number of columns a multiple of " +
          c10::to_string(NUM_ELEM_PER_BYTE));

  // The "fused" representation stores the scale and bias with the
  // row-wise quantized data in one tensor.
  // Since we represent the scale and bias in 16-bit float, we'll use the
  // last 4 bytes of each row for scale (2 bytes) and bias (2 bytes).
  // | ... quantized data ... | scale | bias |
  // |    number_of_columns   |  2B   |  2B  |
  std::vector<int64_t> output_shape = {
      embedding_rows,
      static_cast<std::int64_t>(
          (embedding_cols + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
          2 * sizeof(at::Half))};
  auto output = at::empty(
      output_shape,
      weight_contig.options().dtype(at::kByte),
      weight_contig.suggest_memory_format());
  auto* output_data = output.data_ptr<uint8_t>();

#ifdef USE_FBGEMM
  if (!optimized_qparams) {
    if (weight_contig.scalar_type() == at::ScalarType::Half) {
      const auto weight_data =
          static_cast<fbgemm::float16*>(weight_contig.data_ptr());
      at::parallel_for(
          0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
            fbgemm::FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<
                fbgemm::float16>(
                bit_width,
                weight_data + start_idx * embedding_cols,
                end_idx - start_idx,
                embedding_cols,
                output_data + start_idx * output_shape[1]);
          });
    } else {
      const auto weight_data = weight_contig.data_ptr<float>();
      at::parallel_for(
          0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
            fbgemm::FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<float>(
                bit_width,
                weight_data + start_idx * embedding_cols,
                end_idx - start_idx,
                embedding_cols,
                output_data + start_idx * output_shape[1]);
          });
    }
  } else {
#endif // USE_FBGEMM
    const auto output_columns = output.size(output.dim() - 1);
    const auto float_weight =
        weight_contig.scalar_type() == at::ScalarType::Half
        ? weight_contig.to(at::ScalarType::Float)
        : std::move(weight_contig);
    const auto weight_data = float_weight.data_ptr<float>();
    for (const auto row : c10::irange(embedding_rows)) {
      const float* input_row = weight_data + row * embedding_cols;
      std::uint8_t* output_row = output_data + row * output_columns;

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      float Xmin, Xmax;
      if (optimized_qparams) {
        auto [xmax_tensor, xmin_tensor] = at::choose_qparams_optimized(
            float_weight[row], embedding_cols, nbins, ratio, bit_width);
        TORCH_CHECK(
            xmax_tensor.numel() == 1 && xmin_tensor.numel() == 1,
            "Expected choose_qparams_optimized to return min/max tensors of size 1");
        Xmax = xmax_tensor.item<float>();
        Xmin = xmin_tensor.item<float>();
      } else {
        Xmin = *std::min_element(input_row, input_row + embedding_cols);
        Xmax = *std::max_element(input_row, input_row + embedding_cols);
      }
      Xmin = static_cast<at::Half>(Xmin);
      float range = Xmax - Xmin;
      // Set scale to 1.0f for the corner case of Xmax == Xmin .
      // Any non-zero scale would work because during quantization
      // (X - Xmin) / scale will be 0 for all X unless scale is 0.
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      at::Half scale = range == 0 ? 1.0f : range / ((1 << bit_width) - 1);
      float inverse_scale = scale == 0 ? 1.0f : 1.0f / scale;
      if (scale == 0 || std::isinf(inverse_scale)) {
        // Corner case handling when Xmax == Xmin
        // Any scale would work because X - Xmin will be 0 for all X
        scale = 1.0f;
        inverse_scale = 1.0f;
      }
      // Update the scale and zero_point of each row.
      at::Half* output_row_scale_zp = reinterpret_cast<at::Half*>(
          output_row +
          (embedding_cols + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);

      output_row_scale_zp[0] = scale;
      output_row_scale_zp[1] = Xmin;

      // Pack the weight values.
      for (const auto col : c10::irange(embedding_cols)) {
        float X = input_row[col];
        std::uint8_t quantized = std::max(
            0,
            std::min<int>(
                lrintf((X - Xmin) * inverse_scale), (1 << bit_width) - 1));
        // We pack 2 4-bit values in a byte. Index 0 is packed in the lower
        // 4-bits and index 1 is packed in the upper 4-bits.
        if (col % NUM_ELEM_PER_BYTE == 0) {
          output_row[col / NUM_ELEM_PER_BYTE] = quantized;
        } else {
          output_row[col / NUM_ELEM_PER_BYTE] |=
              (quantized << ((col % NUM_ELEM_PER_BYTE) * bit_width));
        }
      } // embedding_cols
    } // embedding_rows
#ifdef USE_FBGEMM
  }
#endif // USE_FBGEMM

  return output;
}

// Applies 4-bit row-wise quantization by determining the range
// (maximum - minimum) and bias (minimum value) of each row in the input
// matrix, and then scaling each element to an 2-bit number between 0 and
// 15.
// To later de-quantize values, the scale (range / 15) and zero_point
// are stored alongside the data. More precisely, each row first has quantized
// values, and then 2-byte fp16 scale and 2-byte zero_offset.
Tensor qembeddingbag_4bit_prepack(
    const Tensor& weight,
    const bool optimized_qparams,
    const int64_t nbins,
    const double ratio) {
  return _qembeddingbag_nbit_prepack_helper(
      weight, 4 /*bit_width*/, optimized_qparams, nbins, ratio);
}

// Applies 2-bit row-wise quantization by determining the range
// (maximum - minimum) and bias (minimum value) of each row in the input
// matrix, and then scaling each element to an 2-bit number between 0 and
// 3.
// To later de-quantize values, the scale (range / 3) and zero_point
// are stored alongside the data. More precisely, each row first has quantized
// values, and then 2-byte fp16 scale and 2-byte zero_offset.
// TODO() - Add 2Bit Embedding Lookup operator.
Tensor qembeddingbag_2bit_prepack(
    const Tensor& weight,
    const bool optimized_qparams,
    const int64_t nbins,
    const double ratio) {
  return _qembeddingbag_nbit_prepack_helper(
      weight, 2 /*bit_width*/, optimized_qparams, nbins, ratio);
}

class QEmbeddingPackWeights final {
 public:
  static c10::intrusive_ptr<EmbeddingPackedParamsBase> run(at::Tensor weight) {
    return PackedEmbeddingBagWeight::prepack(std::move(weight));
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_prepack"),
      TORCH_FN(qembeddingbag_byte_prepack));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_prepack"),
      TORCH_FN(qembeddingbag_4bit_prepack));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_2bit_prepack"),
      TORCH_FN(qembeddingbag_2bit_prepack));
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_prepack"),
      TORCH_FN(QEmbeddingPackWeights::run));
}


TORCH_LIBRARY_IMPL(quantized, Meta, m) {
  m.impl(
      "quantized::embedding_bag_byte_prepack",
      qembeddingbag_byte_prepack_meta);
}

} // namespace
} // namespace native
} // namespace at

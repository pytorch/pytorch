#pragma once

#include "caffe2/perfkernels/fused_8bit_rowwise_embedding_lookup.h"
#include "fp16_fma.h"
#include "lengths_reducer_ops.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);
C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp_denorms);

namespace caffe2 {

template <
    class Context,
    bool with_weights = 0,
    bool is_mean = 0,
    bool use_acc_fp16 = 0,
    bool use_inv_scale = 0,
    bool use_nnpi_fma = 0,
    bool use_fp16_for_embedding_only = 0,
    bool use_acc_fp32 = 0>
class SparseLengthsFused8BitRowwiseFakeFP16Op final : public Operator<Context> {
 public:
  static_assert(
      !(with_weights && is_mean),
      "Cannot have with_weights and is_mean a the same time");

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  explicit SparseLengthsFused8BitRowwiseFakeFP16Op(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~SparseLengthsFused8BitRowwiseFakeFP16Op() noexcept override {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    const auto& data = Input(DATA);
    const auto& indices = Input(INDICES);
    const auto& lengths = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(indices.dim(), 1, "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS must be a vector");

    const float* weights = nullptr;
    if (with_weights) {
      const auto& weights_input = Input(WEIGHTS);
      CAFFE_ENFORCE_EQ(weights_input.dim(), 1, "WEIGHTS must be a vector");
      CAFFE_ENFORCE_EQ(
          weights_input.numel(),
          indices.numel(),
          "WEIGHTS should have the same length as INDICES.");
      weights = weights_input.template data<float>();
    }

    CAFFE_ENFORCE_GT(data.size(1), 8, "DATA must have more than 8 columns");
    // Subtract 8 from the #columns of data for the 4 bytes for scale and 4
    // bytes for bias that we use in the fused representation (per row).
    const std::vector<int64_t> shape = {lengths.size(0), data.size(1) - 8};
    auto* output = Output(0, shape, at::dtype<float>());

    // Copied from Fused8BitRowwiseEmbeddingLookupGenericSlow in
    // fused_8bit_rowwise_embedding_lookup.cc

    int64_t block_size = output->size(1);
    int64_t output_size = output->size(0);
    int64_t index_size = indices.numel();
    int64_t data_size = data.size(0);
    const uint8_t* input = data.template data<uint8_t>();
    const IndexType* indices_data = indices.template data<IndexType>();
    const int* lengths_data = lengths.template data<int>();
    bool normalize_by_length = is_mean;
    float* out = output->template mutable_data<float>();

    std::vector<float> rowTempSums[2];
    rowTempSums[0].resize(block_size);
    rowTempSums[1].resize(block_size);

    // block_size is the number of elements and fused_block_size is the size of
    // an entire row, including scale and bias.
    const auto scale_bias_offset = 8 / sizeof(uint8_t);
    const int64_t fused_block_size = block_size + scale_bias_offset;
    int64_t current = 0;
    for (int m = 0; m < output_size; ++m) {
      memset(out, 0, sizeof(float) * block_size);
      memset(rowTempSums[0].data(), 0, sizeof(float) * block_size);
      memset(rowTempSums[1].data(), 0, sizeof(float) * block_size);

      if (current + lengths_data[m] > index_size) {
        return false;
      }

      for (int i = 0; i < lengths_data[m]; ++i) {
        int64_t idx = indices_data[current];

        int accIdx = 0;
        // Only do double buffer accumulation when block size is even
        if (use_nnpi_fma && block_size % 2 == 0 && block_size <= 96) {
          accIdx = i % 2;
        }

        if (idx < 0 || idx >= data_size) {
          return false;
        }

        const float* scale_bias = reinterpret_cast<const float*>(
            input + fused_block_size * indices_data[current] + block_size);

        float weight = 1.0f;
        if (weights) {
          weight = weights[current];
          if (!use_fp16_for_embedding_only && !use_acc_fp32) {
            // Fake fp16 rounding of weight
            fbgemm::RoundToFloat16(
                &weight, &weight, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
          }
        }
        float scale = scale_bias[0];
        float bias = scale_bias[1];

        // Vendor might store scale as s' = 1 / s which implies b' = b / s
        // We do      x = x_q * s + b
        // Vendor does x = (x_q + b') / s'
        // Solving these equations yields to the results above
        if (use_inv_scale) {
          constexpr float kEpsilon = 1e-8;
          if (fabs(scale) < kEpsilon) {
            if (scale < 0) {
              scale = -kEpsilon;
            } else {
              scale = kEpsilon;
            }
          }
          scale = 1.0 / (1.0 / scale);
          bias = (bias / scale) * scale;
        }

        if (!use_fp16_for_embedding_only && !use_acc_fp32) {
          // Fake fp16 rounding of scale and bias
          fbgemm::RoundToFloat16(
              &scale, &scale, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
          fbgemm::RoundToFloat16(
              &bias, &bias, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
          scale *= weight;
          // Fake fp16 rounding of scale and bias
          fbgemm::RoundToFloat16(
              &scale, &scale, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
        }

        // Fake fp16 rounding of input/ it is already ints
        std::vector<float> input_rounded(block_size);
        for (int j = 0; j < block_size; ++j) {
          input_rounded[j] =
              input[fused_block_size * indices_data[current] + j];
        }

        if (use_fp16_for_embedding_only) {
          // bias *= weight;

          std::vector<float> product_rounded(block_size);
          TypedAxpy<float, float>(
              block_size, scale, input_rounded.data(), product_rounded.data());

          for (int j = 0; j < block_size; ++j) {
            product_rounded[j] += bias;
          }

          // Fake fp16 rounding of scale x input + bias
          fbgemm::RoundToFloat16(
              reinterpret_cast<const float*>(product_rounded.data()),
              product_rounded.data(),
              block_size,
              FLAGS_caffe2_fbgemm_fake_fp16_clamp,
              FLAGS_caffe2_fbgemm_fake_fp16_clamp_denorms);

          // Accumulate w x (scale x input + bias) to output
          TypedAxpy<float, float>(
              block_size,
              weight,
              reinterpret_cast<const float*>(product_rounded.data()),
              out);

        } else if (use_nnpi_fma) {
          std::vector<float> mScale(block_size, scale);
          std::vector<float> mBias(block_size, bias);
          std::vector<float> mWeight(block_size, weight);

          fake_fp16::fma_fp16(
              block_size,
              mBias.data(),
              mWeight.data(),
              rowTempSums[accIdx].data());

          fake_fp16::fma_fp16(
              block_size,
              mScale.data(),
              input_rounded.data(),
              rowTempSums[accIdx].data());
        } else if (use_acc_fp16) {
          bias *= weight;
          fbgemm::RoundToFloat16(
              &bias, &bias, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

          std::vector<float> product_rounded(block_size);
          TypedAxpy<float, float>(
              block_size, scale, input_rounded.data(), product_rounded.data());

          // Fake fp16 rounding of w x scale x input
          fbgemm::RoundToFloat16(
              reinterpret_cast<const float*>(product_rounded.data()),
              product_rounded.data(),
              block_size,
              FLAGS_caffe2_fbgemm_fake_fp16_clamp);

          for (int j = 0; j < block_size; ++j) {
            product_rounded[j] += bias;
          }
          // Fake fp16 rounding of w x scale x input + w x bias
          fbgemm::RoundToFloat16(
              reinterpret_cast<const float*>(product_rounded.data()),
              product_rounded.data(),
              block_size,
              FLAGS_caffe2_fbgemm_fake_fp16_clamp);

          // Accumulate w x scale x input + w x bias to output
          TypedAxpy<float, float>(
              block_size,
              1.0,
              reinterpret_cast<const float*>(product_rounded.data()),
              out);

          // Fake fp16 rounding of out + (w x scale x input + w x bias)
          fbgemm::RoundToFloat16(
              reinterpret_cast<const float*>(out),
              out,
              block_size,
              FLAGS_caffe2_fbgemm_fake_fp16_clamp);
        } else if (use_acc_fp32) {
          for (int j = 0; j < block_size; ++j) {
            float deqVal = fake_fp16::fmafp32_avx_emulation(
                scale,
                input_rounded[j],
                bias);
            rowTempSums[accIdx][j] = fake_fp16::fmafp32_avx_emulation(
                deqVal,
                weight,
                rowTempSums[accIdx][j]);
          }
        } else {
          bias *= weight;
          fbgemm::RoundToFloat16(
              &bias, &bias, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

          TypedAxpy<float, float>(block_size, scale, input_rounded.data(), out);

          for (int j = 0; j < block_size; ++j) {
            out[j] += bias;
          }
        }
        ++current;
      }

      if (use_nnpi_fma || use_acc_fp32) {
        for (int j = 0; j < block_size; ++j) {
          out[j] = rowTempSums[0][j] + rowTempSums[1][j];
        }
      }

      if (use_nnpi_fma) {
        fbgemm::RoundToFloat16(
            reinterpret_cast<const float*>(out),
            out,
            block_size,
            FLAGS_caffe2_fbgemm_fake_fp16_clamp);
      }

      if (normalize_by_length && lengths_data[m]) {
        float scale = 1.f / lengths_data[m];

        if (!use_fp16_for_embedding_only) {
          // Fake fp16 rounding of scale and out
          fbgemm::RoundToFloat16(
              &scale, &scale, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
          fbgemm::RoundToFloat16(
              reinterpret_cast<const float*>(out),
              out,
              block_size,
              FLAGS_caffe2_fbgemm_fake_fp16_clamp);
        }

        // hack: context is not really used
        math::Scale<float, float, CPUContext>(
            block_size, scale, out, out, nullptr);
      }

      out += block_size;
    }
    return current == index_size;
  }

  enum {
    DATA = 0,
    WEIGHTS = 1,
    INDICES = 1 + with_weights,
    LENGTHS = 2 + with_weights,
  };
};

} // namespace caffe2

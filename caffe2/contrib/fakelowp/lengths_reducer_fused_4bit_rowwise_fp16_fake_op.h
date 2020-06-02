#pragma once

#include <immintrin.h>
#include "caffe2/perfkernels/fused_8bit_rowwise_embedding_lookup.h"
#include "fp16_fma.h"
#include "lengths_reducer_ops.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);
C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp_denorms);

namespace caffe2 {

template <
    class Context,
    bool with_weights = 0,
    bool use_fp16_for_embedding_only = 0>
class SparseLengthsFused4BitRowwiseFakeFP16Op final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  explicit SparseLengthsFused4BitRowwiseFakeFP16Op(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~SparseLengthsFused4BitRowwiseFakeFP16Op() noexcept override {}

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

    CAFFE_ENFORCE_GT(
        data.size(1),
        sizeof(at::Half) * 2,
        "DATA must have more than 4 columns");
    constexpr int NUM_ELEM_PER_BYTE = 2;
    // Subtract 8 from the #columns of data for the 4 bytes for scale and 4
    // bytes for bias that we use in the fused representation (per row).
    const std::vector<int64_t> shape = {
        lengths.size(0),
        static_cast<int64_t>(data.size(1) - 2 * sizeof(at::Half)) *
            NUM_ELEM_PER_BYTE};
    auto* output = Output(0, shape, at::dtype<float>());

    // Copied from Fused8BitRowwiseEmbeddingLookupGenericSlow in
    // fused_8bit_rowwise_embedding_lookup.cc

    int64_t output_block_size = output->size(1);
    CAFFE_ENFORCE_EQ(
        output_block_size % NUM_ELEM_PER_BYTE,
        0,
        "block size must be divisible by 2");
    int64_t input_block_size = output_block_size / NUM_ELEM_PER_BYTE;
    int64_t output_size = output->size(0);
    int64_t index_size = indices.numel();
    int64_t data_size = data.size(0);
    const uint8_t* input = data.template data<uint8_t>();
    const IndexType* indices_data = indices.template data<IndexType>();
    const int* lengths_data = lengths.template data<int>();
    float* out = output->template mutable_data<float>();

    std::vector<float> rowTempSums[2];
    rowTempSums[0].resize(output_block_size);
    rowTempSums[1].resize(output_block_size);

    const auto scale_bias_offset = 2 * sizeof(at::Half);
    const int64_t input_fused_block_size = input_block_size + scale_bias_offset;
    int64_t current = 0;
    for (int m = 0; m < output_size; ++m) {
      if (!use_fp16_for_embedding_only) {
        memset(rowTempSums[0].data(), 0, sizeof(float) * output_block_size);
        memset(rowTempSums[1].data(), 0, sizeof(float) * output_block_size);
      }

      memset(out, 0, sizeof(float) * output_block_size);

      if (current + lengths_data[m] > index_size) {
        return false;
      }

      for (int i = 0; i < lengths_data[m]; ++i) {
        int64_t idx = indices_data[current];

        int accIdx = 0;
        if (output_block_size % 2 == 0 && output_block_size <= 96 &&
            data.size(1) % 2 == 0) {
          accIdx = i % 2;
        }

        if (idx < 0 || idx >= data_size) {
          return false;
        }

        const at::Half* scale_bias = reinterpret_cast<const at::Half*>(
            input + input_fused_block_size * indices_data[current] +
            input_block_size);

        float weight = 1.0f;
        if (weights) {
          weight = weights[current];
          if (!use_fp16_for_embedding_only) {
            // Fake fp16 rounding of weight
            fbgemm::RoundToFloat16(
                &weight, &weight, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
          }
        }
        float scale = scale_bias[0];
        float bias = scale_bias[1];

        if (!use_fp16_for_embedding_only) {
          scale *= weight;
          fbgemm::RoundToFloat16(
              &scale, &scale, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
        }

        // Unpack int4 elements
        std::vector<float> input_rounded(output_block_size);
        int k = 0;
        for (int j = 0; j < input_block_size; j++) {
          input_rounded[k++] =
              input[input_fused_block_size * indices_data[current] + j] & 0x0f;
          input_rounded[k++] =
              input[input_fused_block_size * indices_data[current] + j] >> 4;
        }

        if (use_fp16_for_embedding_only) {
          std::vector<float> product_rounded(output_block_size);
          TypedAxpy<float, float>(
              output_block_size,
              scale,
              input_rounded.data(),
              product_rounded.data());

          for (int j = 0; j < output_block_size; ++j) {
            product_rounded[j] += bias;
          }

          // Fake fp16 rounding of scale x input + bias
          fbgemm::RoundToFloat16(
              reinterpret_cast<const float*>(product_rounded.data()),
              product_rounded.data(),
              output_block_size,
              FLAGS_caffe2_fbgemm_fake_fp16_clamp,
              FLAGS_caffe2_fbgemm_fake_fp16_clamp_denorms);

          // Accumulate w x (scale x input + bias) to output
          TypedAxpy<float, float>(
              output_block_size,
              weight,
              reinterpret_cast<const float*>(product_rounded.data()),
              out);
        } else {
          std::vector<float> product(output_block_size);
          std::vector<float> scalev(output_block_size, scale);
          std::vector<float> mBias(output_block_size, bias);
          std::vector<float> mWeight(output_block_size, weight);

          fake_fp16::fma_fp16(
              output_block_size,
              mBias.data(),
              mWeight.data(),
              rowTempSums[accIdx].data());

          fake_fp16::fma_fp16(
              output_block_size,
              scalev.data(),
              input_rounded.data(),
              rowTempSums[accIdx].data());
        }
        ++current;
      }

      if (!use_fp16_for_embedding_only) {
        for (int j = 0; j < output_block_size; ++j) {
          out[j] = rowTempSums[0][j] + rowTempSums[1][j];
        }
        fbgemm::RoundToFloat16(
            reinterpret_cast<const float*>(out),
            out,
            output_block_size,
            FLAGS_caffe2_fbgemm_fake_fp16_clamp);
      }

      out += output_block_size;
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

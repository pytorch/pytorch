#pragma once

#include <fbgemm/FbgemmConvert.h>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/perfkernels/typed_axpy.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);
C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp_denorms);

namespace caffe2 {

// A templated class that implements SparseLengths[Sum,WeightedSum,Mean].
template <
    class InputTypes, // supported input types, such as TensorTypes<float>
    bool USE_WEIGHT = 0, // Whether it is SparseLengthsWeightedSum
    bool USE_MEAN = 0, // Whether this is SparseLengthsMean
    bool USE_POSITIONAL_WEIGHT = 0,
    bool USE_ACC_FP16 = 0, // Whether use fp16 accumulation
    bool USE_FP16_FOR_EMBEDDING_ONLY =
        0 // Whether use fp16 for embedding entries only
    // USE_WEIGHT = 1 and USE_POSITIONAL_WEIGHT = 1
    // -> SparseLengthsPositionalWeightedSum
    >
class SparseLengthsReductionFakeFp16Op final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  template <class... Args>
  explicit SparseLengthsReductionFakeFp16Op(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...) {
    static_assert(
        !(USE_WEIGHT & USE_MEAN), "Cannot both specify weight and mean.");
  }

  ~SparseLengthsReductionFakeFp16Op() noexcept override {}

  // Currently, we support float and at::Half inputs for input data type, and
  // int32_t and int64_t for the index type.

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(DATA));
  }

  template <typename InputType>
  bool DoRunWithType() {
    return DispatchHelper<TensorTypes2<int32_t, int64_t>, InputType>::call(
        this, Input(INDICES));
  }

  template <typename InputType, typename IndexType>
  bool DoRunWithType2() {
    auto& dataInput = Input(DATA);
    auto& indicesInput = Input(INDICES);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    const int64_t N = dataInput.size(0);
    const int D = dataInput.size_from_dim(1);
    const int64_t M = lengthsInput.size(0);
    const int64_t indices_size = indicesInput.numel();

    auto shape = dataInput.sizes().vec();
    shape[0] = M;
    auto* output = Output(0, shape, at::dtype<float>());
    float* out_data = output->template mutable_data<float>();

    const InputType* in_data = dataInput.template data<InputType>();
    const IndexType* indices = indicesInput.template data<IndexType>();
    const int* lengths = lengthsInput.template data<int>();
    const float* in_weight = nullptr;

    if (USE_WEIGHT) {
      // static if
      auto& weightInput = Input(WEIGHT);
      CAFFE_ENFORCE_EQ(1, weightInput.dim(), "WEIGHT must be a vector");
      if (!USE_POSITIONAL_WEIGHT) {
        CAFFE_ENFORCE_EQ(
            weightInput.numel(),
            indices_size,
            "Weight should have the same length as indices.");
      }
      in_weight = weightInput.template data<float>();
    }

    // Copied from EmbeddingLookupGenericSlow in perfkernels/embedding_lookup.cc
    int64_t block_size = D;
    int64_t output_size = M;
    int64_t index_size = indices_size;
    int64_t data_size = N;
    const InputType* input = in_data;
    const float* weights = in_weight;
    bool normalize_by_lengths = USE_MEAN;
    float* out = out_data;

    int64_t current = 0;
    for (const auto m : c10::irange(output_size)) {
      memset(out, 0, sizeof(float) * block_size);
      if (current + lengths[m] > index_size) {
        return false;
      }
      for (int i = 0; i < lengths[m]; ++i) {
        int64_t idx = indices[current];
        if (idx < 0 || idx >= data_size) {
          return false;
        }

        float w = 1.f;
        if (weights) {
          w = weights[USE_POSITIONAL_WEIGHT ? i : current];
          if (!USE_FP16_FOR_EMBEDDING_ONLY) {
            // Fake fp16 rounding of w
            fbgemm::RoundToFloat16(
                &w, &w, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
          }
        }

        if (USE_FP16_FOR_EMBEDDING_ONLY) {
          std::vector<float> product_rounded(block_size);
          if (std::is_same<InputType, at::Half>::value) {
            TypedAxpy<InputType, float>(
                block_size,
                w,
                input + block_size * indices[current],
                product_rounded.data());
          } else {
            bool is_float = std::is_same<InputType, float>::value;
            assert(is_float);
            // Fake fp16 rounding of input
            std::vector<float> input_rounded(block_size);
            fbgemm::RoundToFloat16(
                reinterpret_cast<const float*>(
                    input + block_size * indices[current]),
                input_rounded.data(),
                block_size,
                FLAGS_caffe2_fbgemm_fake_fp16_clamp,
                FLAGS_caffe2_fbgemm_fake_fp16_clamp_denorms);

            TypedAxpy<float, float>(
                block_size,
                w,
                reinterpret_cast<const float*>(input_rounded.data()),
                product_rounded.data());
          }

          // Accumulate w x input to output
          TypedAxpy<float, float>(
              block_size,
              1.0,
              reinterpret_cast<const float*>(product_rounded.data()),
              out);
        } else if (USE_ACC_FP16) {
          std::vector<float> product_rounded(block_size);
          if (std::is_same<InputType, at::Half>::value) {
            TypedAxpy<InputType, float>(
                block_size,
                w,
                input + block_size * indices[current],
                product_rounded.data());
          } else {
            bool is_float = std::is_same<InputType, float>::value;
            assert(is_float);
            // Fake fp16 rounding of input
            std::vector<float> input_rounded(block_size);
            fbgemm::RoundToFloat16(
                reinterpret_cast<const float*>(
                    input + block_size * indices[current]),
                input_rounded.data(),
                block_size,
                FLAGS_caffe2_fbgemm_fake_fp16_clamp);

            TypedAxpy<float, float>(
                block_size,
                w,
                reinterpret_cast<const float*>(input_rounded.data()),
                product_rounded.data());
          }

          // Fake fp16 rounding of w x input
          fbgemm::RoundToFloat16(
              reinterpret_cast<const float*>(product_rounded.data()),
              product_rounded.data(),
              block_size,
              FLAGS_caffe2_fbgemm_fake_fp16_clamp);

          // Accumulate w x input to output
          TypedAxpy<float, float>(
              block_size,
              1.0,
              reinterpret_cast<const float*>(product_rounded.data()),
              out);

          // Fake fp16 rounding of out + w x input
          fbgemm::RoundToFloat16(
              reinterpret_cast<const float*>(out),
              out,
              block_size,
              FLAGS_caffe2_fbgemm_fake_fp16_clamp);
        } else {
          if (std::is_same<InputType, at::Half>::value) {
            TypedAxpy<InputType, float>(
                block_size, w, input + block_size * indices[current], out);
          } else {
            bool is_float = std::is_same<InputType, float>::value;
            assert(is_float);
            // Fake fp16 rounding of input
            std::vector<float> input_rounded(block_size);
            fbgemm::RoundToFloat16(
                reinterpret_cast<const float*>(
                    input + block_size * indices[current]),
                input_rounded.data(),
                block_size,
                FLAGS_caffe2_fbgemm_fake_fp16_clamp);

            TypedAxpy<float, float>(
                block_size,
                w,
                reinterpret_cast<const float*>(input_rounded.data()),
                out);
          }
        }

        ++current;
      }
      if (normalize_by_lengths && lengths[m]) {
        float scale = 1.f / lengths[m];

        if (!USE_FP16_FOR_EMBEDDING_ONLY) {
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

      if (!USE_FP16_FOR_EMBEDDING_ONLY) {
        // Fake fp16 rounding of out
        fbgemm::RoundToFloat16(
            reinterpret_cast<const float*>(out),
            reinterpret_cast<float*>(out),
            block_size,
            FLAGS_caffe2_fbgemm_fake_fp16_clamp);
      }

      out += block_size;
    }
    return current == index_size;
  }

  enum {
    DATA = 0, // Data input.
    WEIGHT = 1, // Weight input used in SparseLengthsWeightedSum
    INDICES = 1 + USE_WEIGHT, // 1 in SparseLengths[Sum,Mean] and
                              // 2 in SparseLengthsWeightedSum
    LENGTHS = 2 + USE_WEIGHT, // 2 in SparseLengths[Sum, Mean],
                              // 3 in SparseLengthsWeightedSum
  };
};

} // namespace caffe2

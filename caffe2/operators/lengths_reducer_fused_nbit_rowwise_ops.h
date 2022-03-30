#ifndef CAFFE2_OPERATORS_LENGTHS_REDUCER_FUSED_NBIT_ROWWISE_OPS_H_
#define CAFFE2_OPERATORS_LENGTHS_REDUCER_FUSED_NBIT_ROWWISE_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/fused_rowwise_nbit_conversion_ops.h"
#include "caffe2/operators/reducer_functors.h"
#include "caffe2/utils/math.h"
#ifdef USE_FBGEMM
#include "fbgemm/FbgemmEmbedding.h"
#endif

namespace caffe2 {

template <
    int BIT_RATE,
    class Context,
    bool with_weights = false,
    bool is_mean = false>
class SparseLengthsFusedNBitRowwiseOp final : public Operator<Context> {
 public:
  static_assert(
      !(with_weights && is_mean),
      "Cannot have with_weights and is_mean a the same time");

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseLengthsFusedNBitRowwiseOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  ~SparseLengthsFusedNBitRowwiseOp() override {}

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
        sizeof(at::Half) + sizeof(at::Half),
        "DATA must have more than 4 columns");
    static_assert(8 % BIT_RATE == 0, "BIT_RATE must divide 8");
    constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
    // Subtract 4 from the #columns of data for the 2 bytes for fp16 scale and 2
    // byte for bias that we use in the fused representation (per row).
    const std::vector<int64_t> shape = {
        lengths.size(0),
        static_cast<int64_t>(data.size(1) - 2 * sizeof(at::Half)) *
            NUM_ELEM_PER_BYTE};
    auto* output = Output(0, shape, at::dtype<float>());

    int output_size = output->size(0);
    int block_size = output->size(1);
    CAFFE_ENFORCE_EQ(
        block_size % NUM_ELEM_PER_BYTE,
        0,
        "block size must be divisible by " + std::to_string(NUM_ELEM_PER_BYTE));
    int index_size = indices.numel();
    auto data_size = data.size(0);
    const uint8_t* input_data = data.template data<uint8_t>();
    const IndexType* indices_data = indices.template data<IndexType>();
    const int* lengths_data = lengths.template data<int>();
    float* output_data = output->template mutable_data<float>();

#ifdef USE_FBGEMM
    // If this is the first call or block size has changed (should never happen
    // actually), generate a kernel.
    if (block_size != last_block_size) {
      last_block_size = block_size;
      if (std::is_same<IndexType, std::int32_t>::value) {
        kernel32_ = fbgemm::GenerateEmbeddingSpMDMNBit<std::int32_t>(
            BIT_RATE,
            block_size,
            weights != nullptr,
            is_mean,
            /*prefetch distance*/ 8,
            /*is_weight_positional*/ false,
            /*use_offsets*/ false);
      } else {
        CAFFE_ENFORCE((std::is_same<IndexType, std::int64_t>::value));
        kernel64_ = fbgemm::GenerateEmbeddingSpMDMNBit<std::int64_t>(
            BIT_RATE,
            block_size,
            weights != nullptr,
            is_mean,
            /*prefetch distance*/ 8,
            /*is_weight_positional*/ false,
            /*use_offsets*/ false);
      }
    }

    bool success;
    if (std::is_same<IndexType, std::int32_t>::value) {
      success = kernel32_(
          output_size,
          index_size,
          data_size,
          input_data,
          reinterpret_cast<const std::int32_t*>(indices_data),
          lengths_data,
          weights,
          output_data);
    } else {
      success = kernel64_(
          output_size,
          index_size,
          data_size,
          input_data,
          reinterpret_cast<const std::int64_t*>(indices_data),
          lengths_data,
          weights,
          output_data);
    }

    if (success) {
      return true;
    }

    // Error handling
    int64_t current = 0;
    for (const auto m : c10::irange(output_size)) {
      for (int i = 0; i < lengths_data[m]; ++i) {
        CAFFE_ENFORCE_LT(current, index_size);
        IndexType idx = indices_data[current];
        CAFFE_ENFORCE(
            0 <= idx && idx < data_size,
            "Index ",
            current,
            " is out of bounds: ",
            idx,
            ", range 0 to ",
            data_size);
        ++current;
      }
    }
    CAFFE_ENFORCE_EQ(
        current,
        index_size,
        "Your input seems to be incorrect: the sum of lengths values should be "
        "the size of the indices tensor, but it appears not.");

    return false;
#else
    C10_LOG_EVERY_N(WARNING, 10)
        << "Running slow path because FBGEMM is not available";

    int64_t current = 0;
    for (const auto m : c10::irange(output_size)) {
      memset(output_data, 0, block_size * sizeof(float));
      if (current + lengths_data[m] > index_size) {
        return false;
      }
      for (int i = 0; i < lengths_data[m]; ++i, ++current) {
        IndexType idx = indices_data[current];
        if (idx < 0 || idx >= data_size) {
          return false;
        }

        const at::Half* scale_bias = reinterpret_cast<const at::Half*>(
            input_data + (idx + 1) * data.size(1) - 2 * sizeof(at::Half));

        float weight = 1.0f;
        if (with_weights) {
          weight = weights[current];
        }
        const float scale = weight * scale_bias[0];
        const float bias = weight * scale_bias[1];

        for (const auto j : c10::irange(block_size)) {
          uint8_t quantized =
              input_data[idx * data.size(1) + j / NUM_ELEM_PER_BYTE];
          quantized >>= (j % NUM_ELEM_PER_BYTE) * BIT_RATE;
          quantized &= (1 << BIT_RATE) - 1;

          output_data[j] = std::fma(scale, quantized, output_data[j] + bias);
        }
      } // for each i
      if (is_mean && lengths_data[m]) {
        float scale = 1.0f / lengths_data[m];
        for (const auto j : c10::irange(block_size)) {
          output_data[j] *= scale;
        }
      }
      output_data += block_size;
    } // for each m

    return current == index_size;
#endif // USE_FBGEMM
  }

  enum {
    DATA = 0,
    WEIGHTS = 1,
    INDICES = 1 + with_weights,
    LENGTHS = 2 + with_weights,
  };

#ifdef USE_FBGEMM
 private:
  std::int64_t last_block_size{-1};
  fbgemm::EmbeddingSpMDMKernelSignature<std::uint8_t, std::int32_t>::Type
      kernel32_;
  fbgemm::EmbeddingSpMDMKernelSignature<std::uint8_t, std::int64_t>::Type
      kernel64_;
#endif
}; // class SparseLengthsFusedNBitRowwiseOp

class SparseLengthsSumSparseLookupOp final : public Operator<CPUContext> {
 public:
  SparseLengthsSumSparseLookupOp(const OperatorDef& def, Workspace* ws)
      : Operator<CPUContext>(def, ws) {}

  ~SparseLengthsSumSparseLookupOp() override {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    const auto& indices = Input(INDICES);
    const auto& lengths = Input(LENGTHS);
    const auto& compressed_indices_mapping = Input(COMPRESSED_INDICES_MAPPING);
    thread_local static std::vector<float> dummy_weight;
    CAFFE_ENFORCE_EQ(indices.dim(), 1, "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS must be a vector");
    CAFFE_ENFORCE_EQ(
        compressed_indices_mapping.dim(), 1, "LENGTHS must be a vector");
    const int32_t* lengths_data = lengths.template data<int32_t>();
    const IndexType* indices_data = indices.template data<IndexType>();
    const int32_t* compressed_indices_mapping_data =
        compressed_indices_mapping.template data<std::int32_t>();
    dummy_weight.resize(indices.size(0));
    const float* weights = dummy_weight.data();
    bool has_weights = (InputSize() > 3);
    if (has_weights) {
      const auto& weights_input = Input(WEIGHTS);
      CAFFE_ENFORCE_EQ(weights_input.dim(), 1, "WEIGHTS must be a vector");
      CAFFE_ENFORCE_EQ(
          weights_input.numel(),
          indices.numel(),
          "WEIGHTS should have the same length as INDICES.");
      weights = weights_input.template data<float>();
    }

    // Allocate for the max possible size for now and later we may shrink the
    // indices size.
    auto* output_indices =
        Output(INDICES, indices.sizes(), at::dtype<IndexType>());
    auto* output_lengths =
        Output(LENGTHS, lengths.sizes(), at::dtype<int32_t>());
    Tensor* output_weights = nullptr;
    float* output_weights_data = dummy_weight.data();
    if (has_weights) {
      output_weights = Output(2, indices.sizes(), at::dtype<float>());
      output_weights_data = output_weights->template mutable_data<float>();
    }
    int32_t* output_lengths_data =
        output_lengths->template mutable_data<int32_t>();
    IndexType* output_indices_data =
        output_indices->template mutable_data<IndexType>();
    const int32_t output_size = lengths.size(0);
    const IndexType index_size = indices.size(0);
    const IndexType compressed_data_size = compressed_indices_mapping.size(0);
    IndexType current = 0;
    IndexType current_output = 0;
    for (const auto m : c10::irange(output_size)) {
      const auto current_length = lengths_data[m];
      if (current + current_length > index_size) {
        return false;
      }
      int32_t skipped = 0;
      for (const auto i : c10::irange(current_length)) {
        (void)i; // Suppress unused variable warning
        IndexType compressed_idx = indices_data[current];
        if (compressed_idx < 0 || compressed_idx >= compressed_data_size) {
          return false;
        }
        IndexType idx = compressed_indices_mapping_data[compressed_idx];
        if (idx == -1) {
          ++skipped;
        } else {
          output_weights_data[current_output] = weights[current];
          output_indices_data[current_output++] = idx;
        }
        ++current;
      }
      output_lengths_data[m] = current_length - skipped;
    }

    if (current_output < index_size) {
      output_indices->ShrinkTo(current_output);
      if (output_weights) {
        output_weights->ShrinkTo(current_output);
      }
    }
    return true;
  }

  enum {
    INDICES = 0,
    LENGTHS = 1,
    COMPRESSED_INDICES_MAPPING = 2,
    WEIGHTS = 3
  };
};

template <int BIT_RATE, bool with_weights = false, bool is_mean = false>
class SparseLengthsNBitRowwiseSparseOp final : public Operator<CPUContext> {
 public:
  static_assert(
      !(with_weights && is_mean),
      "Cannot have with_weights and is_mean a the same time");

  template<class... Args>
  explicit SparseLengthsNBitRowwiseSparseOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...) {}
  ~SparseLengthsNBitRowwiseSparseOp() override {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    const auto& data = Input(DATA);
    const auto& indices = Input(INDICES);
    const auto& lengths = Input(LENGTHS);
    const auto& compressed_indices_mapping = Input(COMPRESSED_INDICES_MAPPING);
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
        sizeof(at::Half) + sizeof(at::Half),
        "DATA must have more than 4 columns");
    static_assert(8 % BIT_RATE == 0, "BIT_RATE must divide 8");
    constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
    // Subtract 4 (or 8 for BIT_RATE == 8) from the #columns of data for the
    // fp16 (or fp32 for BIT_RATE == 8) scale and bias that we use in the fused
    // representation (per row).
    const std::vector<int64_t> shape = {
        lengths.size(0),
        static_cast<int64_t>(
            data.size(1) -
            2 * (BIT_RATE == 8 ? sizeof(float) : sizeof(at::Half))) *
            NUM_ELEM_PER_BYTE};
    auto* output = Output(0, shape, at::dtype<float>());

    int output_size = output->size(0);
    int block_size = output->size(1);
    CAFFE_ENFORCE_EQ(
        block_size % NUM_ELEM_PER_BYTE,
        0,
        "block size must be divisible by " + std::to_string(NUM_ELEM_PER_BYTE));
    auto data_size = data.size(0);
    int index_size = indices.numel();
    const uint8_t* input_data = data.template data<uint8_t>();
    const IndexType* indices_data = indices.template data<IndexType>();
    const int* lengths_data = lengths.template data<int>();
    float* output_data = output->template mutable_data<float>();
    const std::int32_t* compressed_indices_mapping_data =
        compressed_indices_mapping.template data<std::int32_t>();

    // if compressed_indices_mapping is [0], it is a indicator that
    // we should fallback to normal SLS, which is also a valid fallback if
    // the LUT is pruned.
    const bool fallback_to_no_sparse =
        (compressed_indices_mapping.numel() == 1 &&
         compressed_indices_mapping_data[0] == 0);

#ifdef USE_FBGEMM
    // If this is the first call or block size has changed (should never happen
    // actually), generate a kernel.
    if (block_size != last_block_size) {
      if (!fallback_to_no_sparse) {
        last_block_size = block_size;
        if (std::is_same<IndexType, std::int32_t>::value) {
          if (BIT_RATE == 8) {
            kernel32_ = fbgemm::
                GenerateEmbeddingSpMDMRowWiseSparse<std::uint8_t, std::int32_t>(
                    block_size,
                    weights != nullptr,
                    is_mean,
                    /*prefetch distance*/ 16,
                    /*is_weight_positional*/ false,
                    /*use_offsets*/ false);
          } else {
            kernel32_ =
                fbgemm::GenerateEmbeddingSpMDMNBitRowWiseSparse<std::int32_t>(
                    BIT_RATE,
                    block_size,
                    weights != nullptr,
                    is_mean,
                    /*prefetch distance*/ 16,
                    /*is_weight_positional*/ false,
                    /*use_offsets*/ false);
          }
        } else {
          CAFFE_ENFORCE((std::is_same<IndexType, std::int64_t>::value));
          if (BIT_RATE == 8) {
            kernel64_ = fbgemm::
                GenerateEmbeddingSpMDMRowWiseSparse<std::uint8_t, std::int64_t>(
                    block_size,
                    weights != nullptr,
                    is_mean,
                    /*prefetch distance*/ 16,
                    /*is_weight_positional*/ false,
                    /*use_offsets*/ false);
          } else {
            kernel64_ =
                fbgemm::GenerateEmbeddingSpMDMNBitRowWiseSparse<std::int64_t>(
                    BIT_RATE,
                    block_size,
                    weights != nullptr,
                    is_mean,
                    /*prefetch distance*/ 16,
                    /*is_weight_positional*/ false,
                    /*use_offsets*/ false);
          }
        }
      } else { // fallback_to_no_sparse == true
        last_block_size = block_size;
        if (std::is_same<IndexType, std::int32_t>::value) {
          if (BIT_RATE == 8) {
            kernel32_no_sparse_ =
                fbgemm::GenerateEmbeddingSpMDM<std::uint8_t, std::int32_t>(
                    block_size,
                    with_weights,
                    is_mean,
                    /*prefetch distance*/ 16,
                    /*is_weight_positional*/ false,
                    /*use_offsets*/ false);
          } else {
            kernel32_no_sparse_ =
                fbgemm::GenerateEmbeddingSpMDMNBit<std::int32_t>(
                    BIT_RATE,
                    block_size,
                    weights != nullptr,
                    is_mean,
                    /*prefetch distance*/ 16,
                    /*is_weight_positional*/ false,
                    /*use_offsets*/ false);
          }
        } else {
          CAFFE_ENFORCE((std::is_same<IndexType, std::int64_t>::value));
          if (BIT_RATE == 8) {
            kernel64_no_sparse_ =
                fbgemm::GenerateEmbeddingSpMDM<std::uint8_t, std::int64_t>(
                    block_size,
                    with_weights,
                    is_mean,
                    /*prefetch distance*/ 16,
                    /*is_weight_positional*/ false,
                    /*use_offsets*/ false);
          } else {
            kernel64_no_sparse_ =
                fbgemm::GenerateEmbeddingSpMDMNBit<std::int64_t>(
                    BIT_RATE,
                    block_size,
                    weights != nullptr,
                    is_mean,
                    /*prefetch distance*/ 16,
                    /*is_weight_positional*/ false,
                    /*use_offsets*/ false);
          }
        }
      }
    } // end if (block_size != last_block_size)

    bool success;
    if (!fallback_to_no_sparse) {
      if (std::is_same<IndexType, std::int32_t>::value) {
        success = kernel32_(
            output_size,
            index_size,
            compressed_indices_mapping.size(),
            input_data,
            reinterpret_cast<const std::int32_t*>(indices_data),
            lengths_data,
            weights,
            output_data,
            compressed_indices_mapping_data);
      } else {
        success = kernel64_(
            output_size,
            index_size,
            compressed_indices_mapping.size(),
            input_data,
            reinterpret_cast<const std::int64_t*>(indices_data),
            lengths_data,
            weights,
            output_data,
            compressed_indices_mapping_data);
      }
    } else { // fallback_to_no_sparse == true
      if (std::is_same<IndexType, std::int32_t>::value) {
        success = kernel32_no_sparse_(
            output_size,
            index_size,
            data_size,
            input_data,
            reinterpret_cast<const std::int32_t*>(indices_data),
            lengths_data,
            weights,
            output_data);
      } else {
        success = kernel64_no_sparse_(
            output_size,
            index_size,
            data_size,
            input_data,
            reinterpret_cast<const std::int64_t*>(indices_data),
            lengths_data,
            weights,
            output_data);
      }
    }

    if (success) {
      return true;
    }

    // Error handling
    int64_t current = 0;
    for (const auto m : c10::irange(output_size)) {
      for (int i = 0; i < lengths_data[m]; ++i) {
        CAFFE_ENFORCE_LT(current, index_size);
        IndexType idx = indices_data[current];
        if (!fallback_to_no_sparse) {
          CAFFE_ENFORCE(
              0 <= idx && idx < compressed_indices_mapping.size(),
              "Index ",
              current,
              " is out of bounds: ",
              idx,
              ", range 0 to ",
              compressed_indices_mapping.size());
        } else {
          CAFFE_ENFORCE(
              0 <= idx && idx < data_size,
              "Index ",
              current,
              " is out of bounds: ",
              idx,
              ", range 0 to ",
              data_size);
        }
        ++current;
      }
    }
    CAFFE_ENFORCE_EQ(
        current,
        index_size,
        "Your input seems to be incorrect: the sum of lengths values should be "
        "the size of the indices tensor, but it appears not.");

    return false;
#else
    C10_LOG_EVERY_N(WARNING, 10)
        << "Running slow path because FBGEMM is not available";

    int64_t current = 0;
    for (const auto m : c10::irange(output_size)) {
      memset(output_data, 0, block_size * sizeof(float));
      if (current + lengths_data[m] > index_size) {
        return false;
      }
      for (int i = 0; i < lengths_data[m]; ++i, ++current) {
        IndexType idx;
        if (fallback_to_no_sparse) {
          idx = indices_data[current];
          if (idx < 0 || idx >= data_size) {
            return false;
          }
        } else {
          IndexType uncompressed_idx = indices_data[current];
          if (uncompressed_idx < 0 ||
              uncompressed_idx >= compressed_indices_mapping.size()) {
            return false;
          }
          idx = compressed_indices_mapping_data[uncompressed_idx];
          if (idx == -1) {
            continue;
          }
        }

        const uint8_t* scale_bias = input_data + (idx + 1) * data.size(1) -
            2 * (BIT_RATE == 8 ? sizeof(float) : sizeof(at::Half));

        float weight = 1.0f;
        if (with_weights) {
          weight = weights[current];
        }
        float scale, bias;
        if (BIT_RATE == 8) {
          scale = weight * reinterpret_cast<const float*>(scale_bias)[0];
          bias = weight * reinterpret_cast<const float*>(scale_bias)[1];
        } else {
          scale = weight * reinterpret_cast<const at::Half*>(scale_bias)[0];
          bias = weight * reinterpret_cast<const at::Half*>(scale_bias)[1];
        }

        for (const auto j : c10::irange(block_size)) {
          uint8_t quantized =
              input_data[idx * data.size(1) + j / NUM_ELEM_PER_BYTE];
          quantized >>= (j % NUM_ELEM_PER_BYTE) * BIT_RATE;
          quantized &= (1 << BIT_RATE) - 1;

          output_data[j] = std::fma(scale, quantized, output_data[j] + bias);
        }
      } // for each i
      if (is_mean && lengths_data[m]) {
        float scale = 1.0f / lengths_data[m];
        for (const auto j : c10::irange(block_size)) {
          output_data[j] *= scale;
        }
      }
      output_data += block_size;
    } // for each m

    return current == index_size;
#endif // USE_FBGEMM
  }

  enum {
    DATA = 0,
    WEIGHTS = 1,
    INDICES = 1 + with_weights,
    LENGTHS = 2 + with_weights,
    COMPRESSED_INDICES_MAPPING = 3 + with_weights,
  };

#ifdef USE_FBGEMM
 private:
  std::int64_t last_block_size{-1};
  fbgemm::EmbeddingSpMDMRowWiseSparseKernelSignature<
      std::uint8_t,
      std::int32_t>::Type kernel32_;
  fbgemm::EmbeddingSpMDMRowWiseSparseKernelSignature<
      std::uint8_t,
      std::int64_t>::Type kernel64_;
  fbgemm::EmbeddingSpMDMKernelSignature<std::uint8_t, std::int32_t>::Type
      kernel32_no_sparse_;
  fbgemm::EmbeddingSpMDMKernelSignature<std::uint8_t, std::int64_t>::Type
      kernel64_no_sparse_;
#endif
}; // class SparseLengthsNBitRowwiseSparseOp

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LENGTHS_REDUCER_FUSED_8BIT_ROWWISE_OPS_H_

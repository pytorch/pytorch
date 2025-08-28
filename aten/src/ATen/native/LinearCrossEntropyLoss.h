#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <optional>

namespace at::native {

struct LinearCrossEntropyChecker {
  enum class ChunkingStrategy {
      Unfused,  // Unfused computation
      InputsOnBatch,  // Chunk by inputs on batch dimension
      WeightsOnVocabulary  // Chunk by weights on vocabulary dimension
  };

  static ChunkingStrategy get_chunking_strategy(const std::string_view chunking_strategy) {
    if (chunking_strategy == "unfused") {
      return ChunkingStrategy::Unfused;
    } else if (chunking_strategy == "inputs_on_batch") {
      return ChunkingStrategy::InputsOnBatch;
    } else if (chunking_strategy == "weights_on_vocabulary") {
      return ChunkingStrategy::WeightsOnVocabulary;
    } else {
      TORCH_CHECK(false, "chunking_strategy must be unfused, inputs_on_batch, or weights_on_vocabulary");
    }
  }

  static ChunkingStrategy check(
      Tensor const& input,
      Tensor const& target,
      Tensor const& linear_weight,
      std::optional<Tensor> const& bias,
      std::optional<Tensor> const& cross_entropy_weight,
      std::string_view chunking_strategy,
      double label_smoothing) {    // TODO: documentation
    // TODO: is N dimensions with N>2 useful anywhere?
    auto check_tensor = [](Tensor const& t, const char* arg_name) {
      TORCH_CHECK(
        0 <= t.dim() && t.dim() <= 2,
        "linear_cross_entropy_loss: argument '", arg_name, "' must have 1 or 2 dimensions"
      );
    };

    check_tensor(input, "input");
    check_tensor(target, "target");
    check_tensor(linear_weight, "linear_weight");
    if (bias.has_value()) {
      check_tensor(bias.value(), "bias");
    }
    if (cross_entropy_weight.has_value()) {
      check_tensor(cross_entropy_weight.value(), "cross_entropy_weight");
    }

    auto input_columns = input.sizes()[input.dim() - 1];  // crave back()
    auto weight_rows = linear_weight.sizes()[linear_weight.dim() - 1];
    TORCH_CHECK(
      input_columns == weight_rows,
      "linear_cross_entropy_loss: input.shape()=", input.sizes(),
      " and linear_weight.shape()=", linear_weight.sizes(),
      " are not compatible for matrix multiplication because",
      " input_columns=", input_columns, " != weight_rows=", weight_rows
    );
    TORCH_CHECK(
      label_smoothing >= 0.0 && label_smoothing <= 1.0,
      "linear_cross_entropy_loss: label_smoothing must be between 0.0 and 1.0. Got: ", label_smoothing
    );
    return get_chunking_strategy(chunking_strategy);
  }
};
} // namespace at::native

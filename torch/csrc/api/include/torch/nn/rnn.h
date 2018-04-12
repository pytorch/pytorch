#pragma once

#include <torch/nn/module.h>
#include <torch/nn/rnn.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>

#include <vector>

namespace torch { namespace nn {

// This is largely just a proof-of-concept of the interface.
class LSTM : public torch::nn::CloneableModule<LSTM> {
 public:
  LSTM(long input_features, long state_size)
      : CloneableModule<LSTM>("LSTM"),
        weights_(at::randn(
            torch::CPU(at::kFloat),
            {3 * state_size, input_features * state_size})),
        bias_(at::randn(torch::CPU(at::kFloat), {3 * state_size})) {
    register_parameters({{"weights", weights_}, {"bias", bias_}});
  }

  std::vector<torch::Tensor> forward(
      const std::vector<torch::Tensor>& inputs) override {
    auto& input = inputs[0];
    auto& old_h = inputs[1];
    auto& old_cell = inputs[2];

    auto X = at::cat({old_h, input}, /*dim=*/1);

    auto gate_weights = at::addmm(bias_, X, weights_.transpose(0, 1));
    auto gates = gate_weights.chunk(3, /*dim=*/1);

    auto input_gate = at::sigmoid(gates[0]);
    auto output_gate = at::sigmoid(gates[1]);
    auto candidate_cell = at::elu(gates[2], /*alpha=*/1.0);

    auto new_cell = old_cell + candidate_cell * input_gate;
    auto new_h = at::tanh(new_cell) * output_gate;

    return {new_h, new_cell};
  }

 private:
  Tensor weights_;
  Tensor bias_;
};
}} // namespace torch::nn

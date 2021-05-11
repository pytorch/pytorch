#include "caffe2/operators/inference_lstm_op.h"

namespace caffe2 {
namespace {

bool InferenceLSTMOp::RunOnDevice() {
  auto& _input = Input(0);
  auto& hidden_0 = Input(1);
  auto& hidden_1 = Input(2);
  std::vector<Tensor> params;
  for (int i = 3; i < InputSize(); i++) {
    params.push_back(Input(i).UnsafeSharedInstance());
  }
  auto input = batch_first_ ? transpose(_input, 0, 1, &context_)
                            : _input.UnsafeSharedInstance();

  auto cell_params = gather_params(params, has_biases_, &context_);
  auto results = _lstm_impl(
      input,
      cell_params,
      hidden_0,
      hidden_1,
      num_layers_,
      bidirectional_,
      &context_);

  auto output = copy_ctor(std::get<0>(results));
  if (batch_first_) {
    output = transpose(output, 0, 1, &context_);
  }
  SetOutputTensor(0, copy_ctor(output));
  SetOutputTensor(1, copy_ctor(std::get<1>(results)));
  SetOutputTensor(2, copy_ctor(std::get<2>(results)));
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(InferenceLSTM, InferenceLSTMOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(InferenceLSTM)
    .NumInputs(1, INT_MAX)
    .NumOutputs(3)
    .Output(0, "output", "the output of the last layer of lstm")
    .Output(1, "hidden", "hidden state at t = seq_len")
    .Output(2, "cell", "cell state at t = seq_len")
    .Arg("num_layers", "(*long*): number of layers in the lstm stack")
    .Arg("has_biases", "(*bool*): whether the cells have biases or not")
    .Arg("batch_first", "(*bool*): whether the batch is at dim 0")
    .Arg("bidirectional", "(*bool*): if bidirectional");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(InferenceLSTM);
} // namespace
} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    InferenceLSTM,
    "_caffe2::InferenceLSTM("
      "Tensor[] input_list, "
      "int num_layers, "
      "bool has_biases, "
      "bool batch_first, "
      "bool bidirectional"
    ") -> (Tensor output, Tensor hidden, Tensor cell)",
    caffe2::InferenceLSTMOp);

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

  std::vector<Tensor> allOutputs(OutputSize());
  allOutputs.at(0) = copy_ctor(std::get<0>(results));
  if (batch_first_) {
    allOutputs.at(0) = transpose(allOutputs.at(0), 0, 1, &context_);
  }
  allOutputs.at(1) = copy_ctor(std::get<1>(results));
  allOutputs.at(2) = copy_ctor(std::get<2>(results));
  for (int i = 0; i < OutputSize(); i++) {
    auto output = XOutput(i, allOutputs.at(i).sizes(), dtype<float>());
    context_.CopyItemsSameDevice(
        allOutputs.at(i).dtype(),
        allOutputs.at(i).numel(),
        allOutputs.at(i).template data<float>(),
        output.template mutable_data<float>());
  }
  return true;
}

REGISTER_CPU_OPERATOR(InferenceLSTM, InferenceLSTMOp);
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
NO_GRADIENT(InferenceLSTM);
} // namespace
} // namespace caffe2

C10_REGISTER_CAFFE2_OPERATOR_CPU(
    InferenceLSTM,
    (std::vector<c10::Argument>{
        c10::Argument("input_list", ListType::ofTensors()),
        c10::Argument("num_layers", IntType::get()),
        c10::Argument("has_biases", BoolType::get()),
        c10::Argument("batch_first", BoolType::get()),
        c10::Argument("bidirectional", BoolType::get())}),
    (std::vector<c10::Argument>{c10::Argument("output"),
                                c10::Argument("hidden"),
                                c10::Argument("cell")}),
    caffe2::InferenceLSTMOp);

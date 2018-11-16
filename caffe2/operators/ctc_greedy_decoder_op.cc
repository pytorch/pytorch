#include "caffe2/operators/ctc_greedy_decoder_op.h"

namespace caffe2 {

namespace {

const float* getTensorDataPtr(const Tensor& tensor, int t, int n) {
  const auto dims = tensor.sizes();
  CAFFE_ENFORCE_EQ(dims.size(), 3);
  int offset = (t * dims[1] + n) * dims[2];
  CAFFE_ENFORCE_LT(offset, tensor.numel());
  return tensor.template data<float>() + offset;
}

} // namespace

template <>
bool CTCGreedyDecoderOp<CPUContext>::RunOnDevice() {
  // [max_time_step, batch_size, num_classes]
  auto& inputs = Input(INPUTS);
  // [batch_size]
  auto* output_len = Output(OUTPUT_LEN);
  // [total_decoded_output]
  auto* values = Output(VALUES);

  const auto inputs_dims = inputs.sizes();
  int32_t max_time_step = inputs_dims[0];
  int32_t batch_size = inputs_dims[1];
  int32_t num_classes = inputs_dims[2];
  // [batch_size]
  const int* seq_len_data =
      (InputSize() == 2) ? Input(SEQ_LEN).data<int>() : nullptr;

  vector<int> values_cach;
  output_len->Resize(vector<int64_t>{batch_size});
  int* output_len_data = output_len->template mutable_data<int>();

  for (int32_t i = 0; i < batch_size; ++i) {
    int previous_label = 0, t_dec = 0;
    int32_t seq_len_i = (seq_len_data) ? seq_len_data[i] : max_time_step;
    CAFFE_ENFORCE_LE(seq_len_i, max_time_step);
    for (int32_t t = 0; t < seq_len_i; ++t) {
      auto* prob_data = getTensorDataPtr(inputs, t, i);
      int curr_label =
          std::max_element(prob_data, prob_data + num_classes) - prob_data;
      if (curr_label != 0 &&
          (!merge_repeated_ || (previous_label != curr_label))) {
        t_dec++;
        values_cach.push_back(curr_label);
      }
      previous_label = curr_label;
    }
    output_len_data[i] = t_dec;
  }

  int32_t values_cach_size = values_cach.size();
  values->Resize(vector<int64_t>{values_cach_size});
  int* values_data = values->mutable_data<int>();
  for (int i = 0; i < values_cach.size(); ++i) {
    values_data[i] = values_cach.at(i);
  }
  values_cach.clear();

  return true;
}

REGISTER_CPU_OPERATOR(CTCGreedyDecoder, CTCGreedyDecoderOp<CPUContext>);
OPERATOR_SCHEMA(CTCGreedyDecoder)
    .NumInputs(1, 2)
    .NumOutputs(2)
    .Arg(
        "merge_repeated",
        "When merge_repeated is true, merge repeated classes in output.")
    .SetDoc("Greedy decoder for connectionist temporal classification.")
    .Input(
        0,
        "INPUTS",
        "3D float Tensor sized [max_time, batch_size, num_classes]")
    .Input(
        1,
        "SEQ_LEN",
        "(optional) 1D int vector containing sequence lengths, "
        "having size [batch_size]"
        "seq_len will be set to max_time if not provided")
    .Output(
        0,
        "OUTPUT_LEN",
        "Output_len matrix size (batch). "
        "The row store: [decoded_length]")
    .Output(
        1,
        "VALUES",
        "Values vector, size (total_decoded_outputs). "
        "The vector stores the decoded classes")
    .InheritOnnxSchema();
SHOULD_NOT_DO_GRADIENT(CTCGreedyDecoder);

} // namespace caffe2

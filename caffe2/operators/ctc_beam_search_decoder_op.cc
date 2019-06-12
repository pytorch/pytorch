#include "caffe2/operators/ctc_beam_search_decoder_op.h"

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
bool CTCBeamSearchDecoderOp<CPUContext>::RunOnDevice() {
  // shape: max_activation_length x batch_size x alphabet_size
  auto& inputs = Input(INPUTS);
  // shape: batch_size
  auto* output_len = Output(OUTPUT_LEN);
  // shape: sum over all decoded_length
  auto* values = Output(VALUES);

  const auto inputs_dims = inputs.sizes();
  int32_t max_activation_length = inputs_dims[0];
  int32_t batch_size = inputs_dims[1];
  int32_t alphabet_size = inputs_dims[2];
  // [batch_size]
  const int* seq_len_data =
      (InputSize() == 2) ? Input(SEQ_LEN).data<int>() : nullptr;

  vector<int32_t> values_cache;
  output_len->Resize(vector<int64_t>{batch_size});
  int* output_len_data = output_len->mutable_data<int>();

  for (int32_t i = 0; i < batch_size; ++i) {
    const int32_t activation_length =
        (seq_len_data) ? seq_len_data[i] : max_activation_length;
    std::multimap<float, vector<int32_t>, std::greater<float>> A_next_inv;
    // For a given time step, Pb maps prefixes to the probability of all
    // candidate sequences that end in a blank and Pnb maps prefixes to the
    // probability of all candidate sequences that don't end in a blank.
    vector<std::map<vector<int32_t>, float>> Pb(
        activation_length + 1, std::map<vector<int32_t>, float>());
    vector<std::map<vector<int32_t>, float>> Pnb(
        activation_length + 1, std::map<vector<int32_t>, float>());
    set<vector<int32_t>> A_prev;
    Pb[0][vector<int32_t>()] = 1;
    Pnb[0][vector<int32_t>()] = 0;
    A_prev.insert(vector<int32_t>());

    for (int t = 0; t < activation_length; t++) {
      const float* ctc = getTensorDataPtr(inputs, t, i);

      vector<int32_t> pruned_alpha;
      for (int32_t c = 0; c < alphabet_size; c++) {
        if (ctc[c] > prune_threshold_) {
          pruned_alpha.push_back(c);
        }
      }

      // If the pruned alphabet is empty, don't use pruning.
      if (pruned_alpha.size() == 0) {
        pruned_alpha = vector<int32_t>(alphabet_size);
        std::iota(pruned_alpha.begin(), pruned_alpha.end(), 0);
      }

      for (auto const& l : A_prev) {
        // We skip the code handling the end character from the article since
        // our system does not support an end character.

        for (auto const c : pruned_alpha) {
          // Assumption: blank character always mapped to index 0
          if (c == 0) {
            Pb[t + 1][l] += ctc[c] * (Pb[t][l] + Pnb[t][l]);
          } else {
            vector<int32_t> l_plus = vector<int32_t>(l);
            l_plus.push_back(c);
            if (l.size() > 0 && c == l.back()) {
              Pnb[t + 1][l_plus] += ctc[c] * Pb[t][l];
              Pnb[t + 1][l] += ctc[c] * Pnb[t][l];
            } else {
              Pnb[t + 1][l_plus] += ctc[c] * (Pb[t][l] + Pnb[t][l]);
            }

            if (A_prev.find(l_plus) == A_prev.end()) {
              Pb[t + 1][l_plus] += ctc[0] * (Pb[t][l_plus] + Pnb[t][l_plus]);
              Pnb[t + 1][l_plus] += ctc[c] * Pnb[t][l_plus];
            }
          }
        }
      }

      std::map<vector<int32_t>, float> A_next(Pb[t + 1]);
      for (auto& it : Pnb[t + 1]) {
        A_next[it.first] += it.second;
      }
      A_next_inv.clear();
      for (auto& it : A_next) {
        A_next_inv.insert({it.second, it.first});
      }

      A_prev.clear();
      auto it = A_next_inv.begin();
      for (int j = 0; j < beam_width_; j++) {
        if (it == A_next_inv.end()) {
          break;
        }
        A_prev.insert(it->second);
        it++;
      }
    }

    vector<int32_t> decoded =
        (A_next_inv.empty()) ? vector<int32_t>() : A_next_inv.begin()->second;

    output_len_data[i] = decoded.size();
    values_cache.insert(values_cache.end(), decoded.begin(), decoded.end());
  }

  int32_t cache_size = values_cache.size();
  values->Resize(vector<int64_t>{cache_size});
  int* values_data = values->mutable_data<int>();
  for (int i = 0; i < values_cache.size(); ++i) {
    values_data[i] = values_cache.at(i);
  }
  values_cache.clear();

  return true;
}

REGISTER_CPU_OPERATOR(CTCBeamSearchDecoder, CTCBeamSearchDecoderOp<CPUContext>);
OPERATOR_SCHEMA(CTCBeamSearchDecoder)
    .NumInputs(1, 2)
    .NumOutputs(2)
    .SetDoc(
        "Prefix beam search decoder for connectionist temporal classification.")
    .Arg(
        "beam_width",
        "Maximum number of candidates to carry over to next activation step.")
    .Arg(
        "prune_threshold",
        "Probability threshold below which outputs are ignored.")
    .Input(
        0,
        "INPUTS",
        "3D float Tensor sized [max_activation_length, batch_size, alphabet_size] "
        "of network logits (before softmax application).")
    .Input(
        1,
        "SEQ_LEN",
        "(optional) 1D int vector containing sequence lengths, "
        "having size [batch_size] "
        "seq_len will be set to max_time if not provided.")
    .Output(
        0,
        "OUTPUT_LEN",
        "Output_len matrix size (batch_size). "
        "Each index stores final output length of its corresponding batch item.")
    .Output(
        1,
        "VALUES",
        "Values vector, size (total_decoded_outputs). "
        "The flattened vector of final output sequences, in batch order.")
    .InheritOnnxSchema();
SHOULD_NOT_DO_GRADIENT(CTCBeamSearchDecoder);

} // namespace caffe2

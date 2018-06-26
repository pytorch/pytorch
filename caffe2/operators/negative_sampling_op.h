#ifndef CAFFE2_OPERATORS_NEGATIVE_SAMPLING_OP_H_
#define CAFFE2_OPERATORS_NEGATIVE_SAMPLING_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
template <class Context>
class UniformNegativeSamplingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  UniformNegativeSamplingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        categorical_limit_(
            OperatorBase::GetSingleArgument<int>("categorical_limit", -1)),
        num_negatives_(
            OperatorBase::GetSingleArgument<int>("num_negatives", 3)) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

  template <typename TInd>
  bool DoRunWithType() {
    const auto& indices = Input(INDICES);
    const auto& lengths = Input(LENGTHS);

    auto* output_indices = Output(0);
    auto* output_lengths = Output(1);
    auto* output_labels = Output(2);

    auto output_size = indices.size() * (1 + num_negatives_);
    output_indices->Resize(output_size);
    output_lengths->ResizeLike(lengths);
    output_labels->Resize(output_size);

    const TInd* indices_data = indices.template data<TInd>();
    const int* lengths_data = lengths.template data<int>();
    TInd* output_indices_data = output_indices->template mutable_data<TInd>();
    int* output_lengths_data = output_lengths->template mutable_data<int>();
    int* output_labels_data = output_labels->template mutable_data<int>();
    std::uniform_int_distribution<TInd> distribution(0, categorical_limit_ - 1);

    int out_pos = 0;
    TInd negative_indice;
    for (int i = 0; i < indices.size(); i++) {
      // positive example
      output_indices_data[out_pos] = indices_data[i];
      output_labels_data[out_pos++] = 1.0;

      auto seed = context_.RandGenerator();
      // negative examples
      for (int negative_index = 0; negative_index < num_negatives_;
           negative_index++) {
        negative_indice = indices_data[i];
        while (negative_indice == indices_data[i]) {
          negative_indice = distribution(seed);
        }
        output_indices_data[out_pos] = negative_indice;
        output_labels_data[out_pos++] = 0.0;
      }
    }

    for (int i = 0; i < lengths.size(); i++) {
      output_lengths_data[i] = lengths_data[i] * (1 + num_negatives_);
    }

    return true;
  }

  INPUT_TAGS(INDICES, LENGTHS);

 private:
  int categorical_limit_;
  int num_negatives_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_NEGATIVE_SAMPLING_OP_H_

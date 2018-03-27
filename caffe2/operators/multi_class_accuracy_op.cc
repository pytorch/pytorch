#include "caffe2/operators/multi_class_accuracy_op.h"

namespace caffe2 {

template <>
bool MultiClassAccuracyOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(PREDICTION);
  auto& label = Input(LABEL);
  auto* Y0 = Output(0);
  auto* Y1 = Output(1);
  DCHECK_EQ(X.ndim(), 2);
  // amount, number of instances
  int N = X.dim32(0);
  // dimension, number of classes
  int D = X.dim32(1);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim32(0), N);
  Y0->Resize(D);
  Y1->Resize(D);

  const auto* Xdata = X.data<float>();
  const auto* labeldata = label.data<int>();
  auto* accuracies = Y0->mutable_data<float>();
  auto* amounts = Y1->mutable_data<int>();
  std::fill(accuracies, accuracies + D, 0);
  std::fill(amounts, amounts + D, 0);

  for (int i = 0; i < N; ++i) {
    float maxval = std::numeric_limits<float>::lowest();
    int maxid = 0;
    for (int j = 0; j < D; ++j) {
      if (Xdata[i * D + j] > maxval) {
        maxval = Xdata[i * D + j];
        maxid = j;
      }
    }
    int labelid = labeldata[i];
    DCHECK_LT(labelid, D);
    if (maxid == labelid) {
      accuracies[labelid]++;
    }
    amounts[labelid]++;
  }

  for (int i = 0; i < D; ++i) {
    int amount = amounts[i];
    if (amount) {
      accuracies[i] /= amount;
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(
  MultiClassAccuracy, MultiClassAccuracyOp<float, CPUContext>);

OPERATOR_SCHEMA(MultiClassAccuracy)
  .NumInputs(2)
  .NumOutputs(2)
  .SetDoc(R"DOC(
Respectively compute accuracy score for each class given a number of instances
and predicted scores of each class for each instance.
)DOC")
  .Input(
    0,
    "prediction",
    "2-D float tensor (N,D,) of predicted scores of each class for "
    "each data. N is the number of instances, i.e., batch size. D is number of "
    "possible classes/labels.")
  .Input(
    1,
    "labels",
    "1-D int tensor (N,) of labels for each instance.")
  .Output(
    0,
    "accuracies",
    "1-D float tensor (D,) of accuracy for each class. If a class has no "
    "instance in the batch, its accuracy score is set to zero.")
  .Output(
    1,
    "amounts",
    "1-D int tensor (D,) of number of instances for each class in the batch.");

SHOULD_NOT_DO_GRADIENT(MultiClassAccuracy);
}  // namespace caffe2

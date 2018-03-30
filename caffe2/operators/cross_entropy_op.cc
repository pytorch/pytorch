#include "caffe2/operators/cross_entropy_op.h"

namespace caffe2 {

namespace {

inline float sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * (tgt - (lgt >= 0)) - log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

inline float sigmoid_xent_backward(float lgt, float tgt) {
  return tgt - 1. / (1. + exp(-lgt));
}
}

template <>
bool LabelCrossEntropyOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto* Y = Output(0);
  int N, D;
  if (X.ndim() > 1) {
    N = X.dim32(0);
    D = X.size_from_dim(1);
  } else {
    N = 1;
    D = X.dim32(0);
  }
  CAFFE_ENFORCE(
      (label.ndim() == 1) || (label.ndim() == 2 && label.dim32(1) == 1));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  Y->Resize(N);
  const auto* Xdata = X.data<float>();
  const auto* labelData = label.data<int>();
  auto* Ydata = Y->mutable_data<float>();
  CAFFE_ENFORCE(
      (ConstEigenVectorArrayMap<int>(labelData, N) < D).all() &&
          (ConstEigenVectorArrayMap<int>(labelData, N) >= 0).all(),
      "Label seems to be outside of supported range. Supported labels are in "
      "range [0,",
      D,
      ")");
  for (int i = 0; i < N; ++i) {
    Ydata[i] = -log(std::max(Xdata[i * D + labelData[i]], kLOG_THRESHOLD()));
  }
  return true;
}

template <>
bool SigmoidCrossEntropyWithLogitsOp<float, CPUContext>::RunOnDevice() {
  auto& logits = Input(0);
  auto& targets = Input(1);
  CAFFE_ENFORCE_EQ(logits.dims(), targets.dims());
  const auto inner_size = logits.ndim() > 0 ? logits.dims().back() : 1;
  const auto outer_size = logits.size() / inner_size;

  auto* out = Output(0);
  if (logits.ndim() == 0) {
    out->Resize(std::vector<TIndex>{});
  } else {
    std::vector<TIndex> dims(logits.dims().begin(), logits.dims().end() - 1);
    out->Resize(dims);
  }
  auto* out_ptr = out->mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();

  auto in_idx = 0;
  for (int i = 0; i < outer_size; ++i) {
    float value = 0;
    for (int j = 0; j < inner_size; ++j) {
      value += sigmoid_xent_forward(logits_ptr[in_idx], targets_ptr[in_idx]);
      ++in_idx;
    }
    out_ptr[i] = -value / inner_size;
  }
  return true;
}

template <>
bool SigmoidCrossEntropyWithLogitsGradientOp<float, CPUContext>::RunOnDevice() {
  auto& g = Input(0);
  auto& logits = Input(1);
  auto& targets = Input(2);
  CAFFE_ENFORCE(logits.dims() == targets.dims());
  const auto inner_size = logits.ndim() > 0 ? logits.dims().back() : 1;
  const auto outer_size = logits.size() / inner_size;
  CAFFE_ENFORCE(g.size() == outer_size);

  auto* out = Output(0);
  out->ResizeLike(logits);
  auto* out_ptr = out->mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();
  auto* g_ptr = g.data<float>();

  auto in_idx = 0;
  for (int i = 0; i < outer_size; ++i) {
    auto g_factor = -g_ptr[i] / inner_size;
    for (int j = 0; j < inner_size; ++j) {
      out_ptr[in_idx] = g_factor *
          sigmoid_xent_backward(logits_ptr[in_idx], targets_ptr[in_idx]);
      ++in_idx;
    }
  }
  return true;
}

template <>
bool WeightedSigmoidCrossEntropyWithLogitsOp<float, CPUContext>::RunOnDevice() {
  auto& logits = Input(0);
  auto& targets = Input(1);
  auto& weights = Input(2);
  CAFFE_ENFORCE(logits.dims() == targets.dims());
  CAFFE_ENFORCE(weights.dims() == targets.dims());
  const auto inner_size = logits.ndim() > 0 ? logits.dims().back() : 1;
  const auto outer_size = logits.size() / inner_size;

  auto* out = Output(0);
  if (logits.ndim() == 0) {
    out->Resize(std::vector<TIndex>{});
  } else {
    std::vector<TIndex> dims(logits.dims().begin(), logits.dims().end() - 1);
    out->Resize(dims);
  }
  auto* out_ptr = out->mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();
  auto* weights_ptr = weights.data<float>();

  auto in_idx = 0;
  for (int i = 0; i < outer_size; ++i) {
    float value = 0;
    for (int j = 0; j < inner_size; ++j) {
      value += sigmoid_xent_forward(logits_ptr[in_idx], targets_ptr[in_idx]) *
          weights_ptr[in_idx];
      ++in_idx;
    }
    out_ptr[i] = -value / inner_size;
  }
  return true;
}

template <>
bool WeightedSigmoidCrossEntropyWithLogitsGradientOp<float, CPUContext>::
    RunOnDevice() {
  auto& g = Input(0);
  auto& logits = Input(1);
  auto& targets = Input(2);
  auto& weights = Input(3);
  CAFFE_ENFORCE(logits.dims() == targets.dims());
  CAFFE_ENFORCE(weights.dims() == targets.dims());
  const auto inner_size = logits.ndim() > 0 ? logits.dims().back() : 1;
  const auto outer_size = logits.size() / inner_size;
  CAFFE_ENFORCE(g.size() == outer_size);

  auto* out = Output(0);
  out->ResizeLike(logits);
  auto* out_ptr = out->mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();
  auto* weights_ptr = weights.data<float>();
  auto* g_ptr = g.data<float>();

  auto in_idx = 0;
  for (int i = 0; i < outer_size; ++i) {
    auto g_factor = -g_ptr[i] / inner_size;
    for (int j = 0; j < inner_size; ++j) {
      out_ptr[in_idx] = g_factor *
          sigmoid_xent_backward(logits_ptr[in_idx], targets_ptr[in_idx]) *
          weights_ptr[in_idx];
      ++in_idx;
    }
  }
  return true;
}

template <>
bool LabelCrossEntropyGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  int N, D;
  if (X.ndim() > 1) {
    N = X.dim32(0);
    D = X.size_from_dim(1);
  } else {
    N = 1;
    D = X.dim32(0);
  }
  CAFFE_ENFORCE(
      (label.ndim() == 1) || (label.ndim() == 2 && label.dim32(1) == 1));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  CAFFE_ENFORCE_EQ(dY.ndim(), 1);
  CAFFE_ENFORCE_EQ(dY.dim32(0), N);
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(dX->size(), 0.f, dX->mutable_data<float>(),
                               &context_);
  const float* Xdata = X.data<float>();
  const float* dYdata = dY.data<float>();
  const int* labelData = label.data<int>();
  float* dXdata = dX->mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    dXdata[i * D + labelData[i]] =
        - dYdata[i] / std::max(Xdata[i * D + labelData[i]], kLOG_THRESHOLD());
  }
  return true;
}

template <>
bool MakeTwoClassOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  auto shape = X.dims();
  shape.push_back(2);
  TIndex N = X.size();
  Y->Resize(shape);
  const auto* Xdata = X.data<float>();
  auto* Ydata = Y->mutable_data<float>();
  for (TIndex i = 0; i < N; ++i) {
    DCHECK_GE(Xdata[i], 0.0);
    DCHECK_LE(Xdata[i], 1.0);
    Ydata[i * 2] = 1.0 - Xdata[i];
    Ydata[i * 2 + 1] = Xdata[i];
  }
  return true;
}

template <>
bool MakeTwoClassGradientOp<float, CPUContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto* dX = Output(0);
  auto shape = dY.dims();
  CAFFE_ENFORCE_GE(shape.size(), 1);
  CAFFE_ENFORCE_EQ(shape.back(), 2);
  shape.pop_back();
  dX->Resize(shape);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  TIndex N = dX->size();
  // use eigen?
  for (TIndex i = 0; i < N; ++i) {
    dXdata[i] = dYdata[i * 2 + 1] - dYdata[i * 2];
  }
  return true;
}

template <>
bool CrossEntropyOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto* Y = Output(0);
  int N, D;
  if (X.ndim() > 1) {
    N = X.dim32(0);
    D = X.size_from_dim(1);
  } else {
    N = 1;
    D = X.dim32(0);
  }
  CAFFE_ENFORCE(
      (label.ndim() == 1) || (label.ndim() == 2 && label.dim32(1) == D));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  Y->Resize(vector<TIndex>{N});
  const float* Xdata = X.data<float>();
  const float* labelData = label.data<float>();
  auto* Ydata = Y->mutable_data<float>();
  CAFFE_ENFORCE(
      (ConstEigenArrayMap<float>(labelData, D, N) <= 1.0f).all() &&
          (ConstEigenArrayMap<float>(labelData, D, N) >= 0.0f).all(),
      "Soft label seems incorrect: label value should be a probability ",
      "between 0 and 1.0. You may be using the wrong cross entropy operator; ",
      "use LabelCrossEntropy if the labels are integers whose values are at ",
      "most the number of classes, ",
      D,
      ".");
  EigenArrayMap<float>(Ydata, 1, N) =
      -(ConstEigenArrayMap<float>(labelData, D, N) *
        ConstEigenArrayMap<float>(Xdata, D, N).cwiseMax(kLOG_THRESHOLD()).log())
           .colwise()
           .sum();
  return true;
}

template <>
bool CrossEntropyGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  int N, D;
  if (X.ndim() > 1) {
    N = X.dim32(0);
    D = X.size_from_dim(1);
  } else {
    N = 1;
    D = X.dim32(0);
  }
  CAFFE_ENFORCE(
      (label.ndim() == 1) || (label.ndim() == 2 && label.dim32(1) == D));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  CAFFE_ENFORCE_EQ(dY.ndim(), 1);
  CAFFE_ENFORCE_EQ(dY.dim32(0), N);
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(
    dX->size(), 0.f, dX->mutable_data<float>(), &context_);
  const float* Xdata = X.data<float>();
  const float* dYdata = dY.data<float>();
  const float* labelData = label.data<float>();
  float* dXdata = dX->mutable_data<float>();
  EigenArrayMap<float>(dXdata, D, N) =
      (ConstEigenArrayMap<float>(labelData, D, N) /
       ConstEigenArrayMap<float>(Xdata, D, N).cwiseMax(kLOG_THRESHOLD()))
          .rowwise() *
      (-ConstEigenVectorArrayMap<float>(dYdata, N).transpose());
  return true;
}

REGISTER_CPU_OPERATOR(LabelCrossEntropy,
                      LabelCrossEntropyOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(LabelCrossEntropyGradient,
                      LabelCrossEntropyGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LabelCrossEntropy)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
Operator computes the cross entropy between the input and the label set. In
 practice, it is most commonly used at the end of models, after the SoftMax
 operator and before the AveragedLoss operator. Note that LabelCrossEntropy
 assumes that the label provided is either a 1D array of size N (batch size), or
 a 2D array of size N x 1 (batch size). Each entry in the label vector indicates
 which is the correct class; as such, each entry must be between 0 and D - 1,
 inclusive, where D is the total number of classes. The formula used is:

                            Y[i] = -log(X[i][j])

 where (i, j) is the classifier's prediction of the jth class (the correct one),
 and i is the batch size. Each log has a lower limit for numerical stability.
)DOC")
    .Input(
        0,
        "X",
        "Input blob from the previous layer, which is almost always "
        "the result of a softmax operation; X is a 2D array of size N x D, where N "
        "is the batch size and D is the number of classes")
    .Input(1, "label", "Blob containing the labels used to compare the input")
    .Output(0, "Y", "Output blob after the cross entropy computation");
OPERATOR_SCHEMA(LabelCrossEntropyGradient)
  .NumInputs(3)
  .NumOutputs(1);

class GetLabelCrossEntropyGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "LabelCrossEntropyGradient", "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(LabelCrossEntropy, GetLabelCrossEntropyGradient);

REGISTER_CPU_OPERATOR(MakeTwoClass,
                      MakeTwoClassOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MakeTwoClassGradient,
                      MakeTwoClassGradientOp<float, CPUContext>);

REGISTER_CPU_OPERATOR(
    SigmoidCrossEntropyWithLogits,
    SigmoidCrossEntropyWithLogitsOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SigmoidCrossEntropyWithLogitsGradient,
    SigmoidCrossEntropyWithLogitsGradientOp<float, CPUContext>);

REGISTER_CPU_OPERATOR(
    WeightedSigmoidCrossEntropyWithLogits,
    WeightedSigmoidCrossEntropyWithLogitsOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    WeightedSigmoidCrossEntropyWithLogitsGradient,
    WeightedSigmoidCrossEntropyWithLogitsGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(MakeTwoClass)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef& /* unused */, const vector<TensorShape>& in) {
          vector<TensorShape> out(1);
          out[0].add_dims(in[0].dims(0));
          out[0].add_dims(2);
          return out;
        })
    .SetDoc(R"DOC(
Given a vector of probabilities, this operator transforms this into a 2-column
 matrix with complimentary probabilities for binary classification. In explicit
 terms, given the vector X, the output Y is vstack(1 - X, X).
  )DOC")
    .Input(0, "X", "Input vector of probabilities")
    .Output(
        0,
        "Y",
        "2-column matrix with complimentary probabilities of X for "
        "binary classification");

OPERATOR_SCHEMA(MakeTwoClassGradient)
  .NumInputs(1)
  .NumOutputs(1);

OPERATOR_SCHEMA(SigmoidCrossEntropyWithLogits)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
Given two matrices logits and targets, of same shape,
(batch_size, num_classes), computes the sigmoid cross entropy between the two.
Returns a tensor of shape (batch_size,) of losses for each example.
)DOC")
    .Input(0, "logits", "matrix of logits for each example and class.")
    .Input(1, "targets", "matrix of targets, same shape as logits.")
    .Output(0, "xentropy", "Vector with the total xentropy for each example.");

OPERATOR_SCHEMA(SigmoidCrossEntropyWithLogitsGradient)
    .NumInputs(3)
    .NumOutputs(1);

OPERATOR_SCHEMA(WeightedSigmoidCrossEntropyWithLogits)
    .NumInputs(3)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
Given three matrices: logits, targets, weights, all of the same shape,
(batch_size, num_classes), computes the weighted sigmoid cross entropy between
logits and targets. Specifically, at each position r,c, this computes
weights[r, c] * crossentropy(sigmoid(logits[r, c]), targets[r, c]), and then
averages over each row.
Returns a tensor of shape (batch_size,) of losses for each example.
)DOC")
    .Input(0, "logits", "matrix of logits for each example and class.")
    .Input(1, "targets", "matrix of targets, same shape as logits.")
    .Input(2, "weights", "matrix of weights, same shape as logits.")
    .Output(0, "xentropy", "Vector with the total xentropy for each example.");

OPERATOR_SCHEMA(WeightedSigmoidCrossEntropyWithLogitsGradient)
    .NumInputs(4)
    .NumOutputs(1);

struct GetMakeTwoClassGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "MakeTwoClassGradient",
        "",
        vector<string>{GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(MakeTwoClass, GetMakeTwoClassGradient);

struct GetSigmoidCrossEntropyWithLogitsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SigmoidCrossEntropyWithLogitsGradient",
        "",
        vector<string>{GO(0), I(0), I(1)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(
    SigmoidCrossEntropyWithLogits,
    GetSigmoidCrossEntropyWithLogitsGradient);

struct GetWeightedSigmoidCrossEntropyWithLogitsGradient
    : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "WeightedSigmoidCrossEntropyWithLogitsGradient",
        "",
        vector<string>{GO(0), I(0), I(1), I(2)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(
    WeightedSigmoidCrossEntropyWithLogits,
    GetWeightedSigmoidCrossEntropyWithLogitsGradient);

REGISTER_CPU_OPERATOR(CrossEntropy,
                      CrossEntropyOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(CrossEntropyGradient,
                      CrossEntropyGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(CrossEntropy)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
Operator computes the cross entropy between the input and the label set. In
 practice, it is most commonly used at the end of models, after the SoftMax
 operator and before the AveragedLoss operator. Note that CrossEntropy
 assumes that the soft labels provided is a 2D array of size N x D
 (batch size x number of classes). Each entry in the 2D label corresponds to
 the soft label for the input, where each element represents the correct
 probability of the class being selected. As such, each element must be between
 0 and 1, and all elements in an entry must sum to 1. The formula used is:

                Y[i] = sum_j (label[i][j] * log(X[i][j]))

 where (i, j) is the classifier's prediction of the jth class (the correct one),
 and i is the batch size. Each log has a lower limit for numerical stability.
)DOC")
    .Input(
        0,
        "X",
        "Input blob from the previous layer, which is almost always "
        "the result of a softmax operation; X is a 2D array of size N x D, where N "
        "is the batch size and D is the number of classes")
    .Input(1, "label", "Blob containing the labels used to compare the input")
    .Output(0, "Y", "Output blob after the cross entropy computation");
OPERATOR_SCHEMA(CrossEntropyGradient)
  .NumInputs(3)
  .NumOutputs(1);

class GetCrossEntropyGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CrossEntropyGradient", "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(CrossEntropy, GetCrossEntropyGradient);

}  // namespace caffe2

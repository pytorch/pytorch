#include "caffe2/operators/cross_entropy_op.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

namespace {

inline float sigmoid_xent_forward(float lgt, float tgt) {
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return lgt * (tgt - (lgt >= 0)) - log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

inline float sigmoid_xent_backward(float lgt, float tgt) {
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return tgt - 1. / (1. + exp(-lgt));
}

inline float sigmoid_partition(float lgt) {
  // computes log(1 + exp(lgt)) with only exp(x) function when x >= 0
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return lgt * (lgt >= 0) + log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

inline float sigmoid_xent_forward_with_log_d_trick(float lgt, float tgt) {
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return (2 * tgt - 1.) * (lgt - sigmoid_partition(lgt));
}

inline float sigmoid_xent_backward_with_log_d_trick(float lgt, float tgt) {
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return (2 * tgt - 1.) / (1. + exp(lgt));
}

inline float unjoined_sigmoid_xent_forward(float lgt, float tgt) {
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return lgt * tgt + (tgt - 1) * lgt * (lgt >= 0) -
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      (1 - tgt) * log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

inline float unjoined_sigmoid_xent_backward(float lgt, float tgt) {
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return tgt - (1. - tgt) / (1. + exp(-lgt));
}

} // namespace

template <>
bool LabelCrossEntropyOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int N, D;
  if (X.dim() > 1) {
    N = X.dim32(0);
    D = X.size_from_dim(1);
  } else {
    N = 1;
    D = X.dim32(0);
  }
  CAFFE_ENFORCE(
      (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == 1));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  auto* Y = Output(0, {N}, at::dtype<float>());
  const auto* Xdata = X.data<float>();
  const auto* labelData = label.data<int>();
  auto* Ydata = Y->template mutable_data<float>();
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
  CAFFE_ENFORCE_EQ(logits.sizes(), targets.sizes());
  const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
  const auto outer_size = logits.numel() / inner_size;

  std::vector<int64_t> dims;
  if (logits.dim() != 0) {
    dims =
        std::vector<int64_t>(logits.sizes().begin(), logits.sizes().end() - 1);
  }
  auto* out = Output(0, dims, at::dtype<float>());
  auto* out_ptr = out->template mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();

  auto in_idx = 0;
  for (int i = 0; i < outer_size; ++i) {
    float value = 0;
    for (int j = 0; j < inner_size; ++j) {
      if (unjoined_lr_loss_) {
        value += unjoined_sigmoid_xent_forward(
            logits_ptr[in_idx], targets_ptr[in_idx]);
      } else {
        value +=
            (log_D_trick_ ? sigmoid_xent_forward_with_log_d_trick(
                                logits_ptr[in_idx], targets_ptr[in_idx])
                          : sigmoid_xent_forward(
                                logits_ptr[in_idx], targets_ptr[in_idx]));
      }
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
  CAFFE_ENFORCE(logits.sizes() == targets.sizes());
  const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
  const auto outer_size = logits.numel() / inner_size;
  CAFFE_ENFORCE(g.numel() == outer_size);

  auto* out = Output(0, logits.sizes(), at::dtype<float>());
  auto* out_ptr = out->template mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();
  auto* g_ptr = g.data<float>();

  auto in_idx = 0;
  for (int i = 0; i < outer_size; ++i) {
    auto g_factor = -g_ptr[i] / inner_size;
    for (int j = 0; j < inner_size; ++j) {
      if (unjoined_lr_loss_) {
        out_ptr[in_idx] = g_factor *
            unjoined_sigmoid_xent_backward(
                              logits_ptr[in_idx], targets_ptr[in_idx]);
      } else {
        out_ptr[in_idx] = g_factor *
            (log_D_trick_ ? sigmoid_xent_backward_with_log_d_trick(
                                logits_ptr[in_idx], targets_ptr[in_idx])
                          : sigmoid_xent_backward(
                                logits_ptr[in_idx], targets_ptr[in_idx]));
      }
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
  CAFFE_ENFORCE(logits.sizes() == targets.sizes());
  CAFFE_ENFORCE(weights.sizes() == targets.sizes());
  const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
  const auto outer_size = logits.numel() / inner_size;

  std::vector<int64_t> dims;
  if (logits.dim() != 0) {
    dims =
        std::vector<int64_t>(logits.sizes().begin(), logits.sizes().end() - 1);
  }

  auto* out = Output(0, dims, at::dtype<float>());
  auto* out_ptr = out->template mutable_data<float>();

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
  CAFFE_ENFORCE(logits.sizes() == targets.sizes());
  CAFFE_ENFORCE(weights.sizes() == targets.sizes());
  const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
  const auto outer_size = logits.numel() / inner_size;
  CAFFE_ENFORCE(g.numel() == outer_size);

  auto* out = Output(0, logits.sizes(), at::dtype<float>());
  auto* out_ptr = out->template mutable_data<float>();

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

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int N, D;
  if (X.dim() > 1) {
    N = X.dim32(0);
    D = X.size_from_dim(1);
  } else {
    N = 1;
    D = X.dim32(0);
  }
  CAFFE_ENFORCE(
      (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == 1));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  CAFFE_ENFORCE_EQ(dY.dim(), 1);
  CAFFE_ENFORCE_EQ(dY.dim32(0), N);
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  math::Set<float, CPUContext>(
      dX->numel(), 0.f, dX->template mutable_data<float>(), &context_);
  const float* Xdata = X.data<float>();
  const float* dYdata = dY.data<float>();
  const int* labelData = label.data<int>();
  float* dXdata = dX->template mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    dXdata[i * D + labelData[i]] =
        -dYdata[i] / std::max(Xdata[i * D + labelData[i]], kLOG_THRESHOLD());
  }
  return true;
}

template <>
bool MakeTwoClassOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);

  auto shape = X.sizes().vec();
  shape.push_back(2);
  int64_t N = X.numel();
  auto* Y = Output(0, shape, at::dtype<float>());
  const auto* Xdata = X.data<float>();
  auto* Ydata = Y->template mutable_data<float>();
  for (int64_t i = 0; i < N; ++i) {
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

  auto shape = dY.sizes().vec();
  CAFFE_ENFORCE_GE(shape.size(), 1);
  CAFFE_ENFORCE_EQ(shape.back(), 2);
  shape.pop_back();
  auto* dX = Output(0, shape, at::dtype<float>());
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();
  int64_t N = dX->numel();
  // use eigen?
  for (int64_t i = 0; i < N; ++i) {
    dXdata[i] = dYdata[i * 2 + 1] - dYdata[i * 2];
  }
  return true;
}

template <>
bool CrossEntropyOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int N, D;
  if (X.dim() > 1) {
    N = X.dim32(0);
    D = X.size_from_dim(1);
  } else {
    N = 1;
    D = X.dim32(0);
  }
  CAFFE_ENFORCE(
      (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == D));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  auto* Y = Output(0, vector<int64_t>{N}, at::dtype<float>());
  const float* Xdata = X.data<float>();
  const float* labelData = label.data<float>();
  auto* Ydata = Y->template mutable_data<float>();
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

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int N, D;
  if (X.dim() > 1) {
    N = X.dim32(0);
    D = X.size_from_dim(1);
  } else {
    N = 1;
    D = X.dim32(0);
  }
  CAFFE_ENFORCE(
      (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == D));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  CAFFE_ENFORCE_EQ(dY.dim(), 1);
  CAFFE_ENFORCE_EQ(dY.dim32(0), N);
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  math::Set<float, CPUContext>(
      dX->numel(), 0.f, dX->template mutable_data<float>(), &context_);
  const float* Xdata = X.data<float>();
  const float* dYdata = dY.data<float>();
  const float* labelData = label.data<float>();
  float* dXdata = dX->template mutable_data<float>();
  EigenArrayMap<float>(dXdata, D, N) =
      (ConstEigenArrayMap<float>(labelData, D, N) /
       ConstEigenArrayMap<float>(Xdata, D, N).cwiseMax(kLOG_THRESHOLD()))
          .rowwise() *
      (-ConstEigenVectorArrayMap<float>(dYdata, N).transpose());
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    LabelCrossEntropy,
    LabelCrossEntropyOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    LabelCrossEntropyGradient,
    LabelCrossEntropyGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(LabelCrossEntropy)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
This operator computes the cross entropy between a $NxD$ dimensional input data tensor $X$  and a one dimensional input label tensor $label$. The op produces a single length $N$ output tensor $Y$. Here, $N$ is considered the batch size and $D$ is the size of each element in the batch. In practice, it is most commonly used at the end of models as a part of the loss computation, after the SoftMax operator and before the AveragedLoss operator. The cross entropy operation is defined as follows

$$Y_i = -log(X_{ij})$$

where ($i$, $j$) is the classifier's prediction of the $j$th class (the correct one), and $i$ is the batch size. Each log has a lower limit for numerical stability.

The difference between *LabelCrossEntropy* and *CrossEntropy* is how the labels are specified. Here, the labels are a length $N$ list of integers, whereas in CrossEntropy the labels are a $NxD$ dimensional matrix of one hot label vectors. However, the results of computation should be the same, as shown in the two examples where ($i$, $j$) is the classifier's prediction of the $j$th class (the correct one), and $i$ is the batch size. Each log has a lower limit for numerical stability.

Github Links:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LabelCrossEntropy",
    ["X", "label"],
    ["Y"]
)

// Create X: Sample softmax output for 5-class model
X = np.array([[.01, .05, .02, .02, .9],[.03, .1, .42, .05, .4]])
print("X:\n",X)

// Create label: Sample 1-hot ground truth label vectors
label = np.array([4,2])
print("label:\n",label)

// Feed X & label into workspace
workspace.FeedBlob("X", X.astype(np.float32))
workspace.FeedBlob("label", label.astype(np.int32))

// Run op
workspace.RunOperatorOnce(op)

// Collect Output
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[0.01 0.05 0.02 0.02 0.9 ]
 [0.03 0.1  0.42 0.05 0.4 ]]
label:
 [4 2]
Y:
 [0.10536055 0.8675006 ]

```

</details>


)DOC")
    .Input(
        0,
        "X",
        "Input tensor which is almost always the result of a softmax operation. $X$ is a 2D array of size $NxD$, where $N$ is the batch size and $D$ is the number of classes.")
    .Input(
        1,
        "label",
        "Blob containing the labels used to compare the input. $label$ is a length $N$ list of integers, where each element is the integer label for the $n$th element of the batch.")
    .Output(
        0,
        "Y",
        "Output blob from the cross entropy computation. $Y$ is 1D length $N$ tensor.");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(LabelCrossEntropyGradient).NumInputs(3).NumOutputs(1);

class GetLabelCrossEntropyGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "LabelCrossEntropyGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(LabelCrossEntropy, GetLabelCrossEntropyGradient);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(MakeTwoClass, MakeTwoClassOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MakeTwoClassGradient,
    MakeTwoClassGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    SigmoidCrossEntropyWithLogits,
    SigmoidCrossEntropyWithLogitsOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    SigmoidCrossEntropyWithLogitsGradient,
    SigmoidCrossEntropyWithLogitsGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    WeightedSigmoidCrossEntropyWithLogits,
    WeightedSigmoidCrossEntropyWithLogitsOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    WeightedSigmoidCrossEntropyWithLogitsGradient,
    WeightedSigmoidCrossEntropyWithLogitsGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MakeTwoClass)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MakeTwoClassGradient).NumInputs(1).NumOutputs(1);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SigmoidCrossEntropyWithLogits)
    .Arg("log_D_trick", R"DOC(
default is false; if enabled, will use the log d trick to avoid the vanishing
gradients early on; see Goodfellow et. al (2014)
)DOC")
    .Arg("unjoined_lr_loss", R"DOC(
default is false; if enabled, the model will be allowed to train on an unjoined
dataset, where some examples might be false negative and might appear
in the dataset later as (true) positive example.
)DOC")
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SigmoidCrossEntropyWithLogitsGradient)
    .NumInputs(3)
    .NumOutputs(1);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(
    WeightedSigmoidCrossEntropyWithLogits,
    GetWeightedSigmoidCrossEntropyWithLogitsGradient);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(CrossEntropy, CrossEntropyOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    CrossEntropyGradient,
    CrossEntropyGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(CrossEntropy)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
This operator computes the cross entropy between a $NxD$ dimensional input data tensor $X$  and a $NxD$ dimensional input label tensor $label$. The op produces a single length $N$ output tensor $Y$. Here, $N$ is considered the batch size and $D$ is the size of each element in the batch. In practice, it is most commonly used at the end of models as a part of the loss computation, after the SoftMax operator and before the AveragedLoss operator. The cross entropy operation is defined as follows

$$Y_i = \sum_j (label_{ij} * log(X_{ij}))$$

where ($i$, $j$) is the classifier's prediction of the $j$th class (the correct one), and $i$ is the batch size. Each log has a lower limit for numerical stability.

Github Links:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "CrossEntropy",
    ["X", "label"],
    ["Y"]
)

// Create X: Sample softmax output for 5-class model
X = np.array([[.01, .05, .02, .02, .9],[.03, .1, .42, .05, .4]])
print("X:\n",X)

// Create label: Sample 1-hot ground truth label vectors
label = np.array([[0.,0.,0.,0.,1.],[0.,0.,1.,0.,0.]])
print("label:\n",label)

// Feed X & label into workspace
workspace.FeedBlob("X", X.astype(np.float32))
workspace.FeedBlob("label", label.astype(np.float32))

// Run op
workspace.RunOperatorOnce(op)

// Collect Output
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[0.01 0.05 0.02 0.02 0.9 ]
 [0.03 0.1  0.42 0.05 0.4 ]]
label:
 [[0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0.]]
Y:
 [0.10536055 0.8675006 ]

```

</details>


)DOC")
    .Input(
        0,
        "X",
        "Input tensor which is almost always the result of a softmax operation. $X$ is a 2D array of size $NxD$, where $N$ is the batch size and $D$ is the number of classes.")
    .Input(
        1,
        "label",
        "Blob containing the labels used to compare the input. $label$ is the same shape as $X$.")
    .Output(
        0,
        "Y",
        "Output blob from the cross entropy computation. $Y$ is 1D length $N$ tensor.");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(CrossEntropyGradient).NumInputs(3).NumOutputs(1);

class GetCrossEntropyGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CrossEntropyGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(CrossEntropy, GetCrossEntropyGradient);

} // namespace caffe2

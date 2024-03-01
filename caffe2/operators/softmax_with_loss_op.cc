#include "caffe2/operators/softmax_with_loss_op.h"

#include <vector>

#include "caffe2/operators/softmax_utils.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SoftmaxWithLoss, SoftmaxWithLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SoftmaxWithLossGradient,
    SoftmaxWithLossGradientOp<float, CPUContext>);

// Input: X (logits), T (labels); Output: P (probs), Y
OPERATOR_SCHEMA(SoftmaxWithLoss)
    .NumInputs(2, 3)
    .NumOutputs({2, 3})
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto axis = helper.GetSingleArgument<int32_t>("axis", 1);

      vector<TensorShape> out(2);

      auto logits = in[0]; // Tensor with Shape [batch_size, num_classes]
      auto labels = in[1]; // Tensor with shape [batch_size, ]
      const auto canonical_axis =
          canonical_axis_index_(axis, logits.dims().size());
      const int batch_size =
          size_to_dim_(canonical_axis, GetDimsVector(logits));
      const int num_classes =
          size_from_dim_(canonical_axis, GetDimsVector(logits));

      out[0].set_data_type(logits.data_type());
      out[0].add_dims(batch_size);
      out[0].add_dims(num_classes);

      return out;
    })
    .SetDoc(R"DOC(
Combined Softmax and Cross-Entropy loss operator. The operator first computes the softmax normalized values for each layer in the batch of the given input, then computes cross-entropy loss. This operator is numerically more stable than separate `Softmax` and `CrossEntropy` ops. The inputs are a 2-D tensor `logits` of size (batch_size x input_feature_dimensions), which represents the unscaled log probabilities, and a 1-dimensional integer `labels` tensor for ground truth. An optional third input blob (`weight_tensor`) can be used to weight the samples for the loss, which is useful if the training set is unbalanced. This operator outputs a `softmax` tensor which contains the probability for each label for each example (same shape is `logits` input), and a scalar `loss` value, which is the averaged cross-entropy loss between the softmax probabilities and the ground truth values. Use parameter `label_prob`=1 to enable inputting labels as a probability distribution.

Softmax cross-entropy loss function:

$$loss(x, class) = -\log{\biggl(\frac{\exp(x[class])}{\sum_{j} \exp(x[j])}\biggr)} = -x[class] + \log{\biggl(\sum_{j} \exp(x[j])\biggr)}$$

or if the `weight_tensor` has been passed:

$$loss(x, class) = weight[class]\biggl(-x[class] + \log{\biggl(\sum_{j} \exp(x[j])\biggr)}\biggr)$$

The `logits` input does not need to explicitly be a 2D vector; rather, it will be coerced into one. For an arbitrary n-dimensional tensor `X` in $[a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}]$, where k is the `axis` provided, then `X` will be coerced into a 2-dimensional tensor with dimensions $[(a_0 * ... * a_{k-1}), (a_k * ... * a_{n-1})]$. For the default case where `axis`=1, the `X` tensor will be coerced into a 2D tensor of dimensions $[a_0, (a_1 * ... * a_{n-1})]$, where $a_0$ is often the batch size. In this situation, we must have $a_0 = N$ and $a_1 * ... * a_{n-1} = D$. Each of these dimensions must be matched correctly, or else the operator will throw errors.

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/softmax_with_loss_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "SoftmaxWithLoss",
    ["logits", "labels"],
    ["softmax", "avgloss"]
)

workspace.FeedBlob("logits", np.random.randn(1, 5).astype(np.float32))
workspace.FeedBlob("labels", np.asarray([4]).astype(np.int32))
print("logits:", workspace.FetchBlob("logits"))
print("labels:", workspace.FetchBlob("labels"))
workspace.RunOperatorOnce(op)
print("softmax:", workspace.FetchBlob("softmax"))
print("avgloss:", workspace.FetchBlob("avgloss"))

```

**Result**

```

logits: [[-0.3429451  -0.80375195  0.23104447  1.4569176  -0.5268362 ]]
labels: [4]
softmax: [[0.09721052 0.0613179  0.17258129 0.58800864 0.0808817 ]]
avgloss: 2.5147676

```

</details>

<details>

<summary> <b>Example 2</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "SoftmaxWithLoss",
    ["logits", "labels"],
    ["softmax", "avgloss"],
    scale=5.0
)

workspace.FeedBlob("logits", np.asarray([[.1, .4, .7, 1.5, .2]]).astype(np.float32))
workspace.FeedBlob("labels", np.asarray([4]).astype(np.int32))
print("logits:", workspace.FetchBlob("logits"))
print("labels:", workspace.FetchBlob("labels"))
workspace.RunOperatorOnce(op)
print("softmax:", workspace.FetchBlob("softmax"))
print("avgloss:", workspace.FetchBlob("avgloss"))

```

**Result**

```

logits: [[0.1 0.4 0.7 1.5 0.2]]
labels: [4]
softmax: [[0.10715417 0.144643   0.19524762 0.4345316  0.11842369]]
avgloss: 10.667433

```

</details>

)DOC")
    .Arg(
        "label_prob",
        "*(type: int; default: 0)* Setting to 1 enables inputting labels as probability distribution.")
    .Arg(
        "axis",
        "*(type: int; default: 1)* Axis of the inputs when coerced to 2D.")
    .Arg(
        "scale",
        "*(type: float)* Average loss output scaling factor (must be >= 0).")
    .Arg(
        "order",
        "*(type: string; default: 'NCHW')* Order of blob dimensions (only 'NCHW' is supported currently).")
    .Input(0, "logits", "*(type: Tensor`<float>`)* Input tensor.")
    .Input(1, "labels", "*(type: Tensor`<float>`)* Ground truth label tensor.")
    .Input(
        2,
        "weight_tensor",
        "*(type: Tensor`<float>`)* [OPTIONAL] Blob used to weight the samples for the loss.")
    .Output(0, "softmax", "*(type: Tensor`<float>`)* Softmax output tensor.")
    .Output(1, "loss", "*(type: float)* Averaged cross-entropy loss output.");

// Input: X, T, P, dY; Output: dX
OPERATOR_SCHEMA(SoftmaxWithLossGradient).NumOutputs(1);

#define DONT_CARE (-1)

template <>
bool SoftmaxWithLossOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0); // Logits
  auto& T = Input(1); // Labels / targets

  const auto canonical_axis = X.canonical_axis_index(axis_);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);
  auto* P =
      Output(0, X.sizes(), at::dtype<float>()); // Probabilities from softmax

  float* Pdata = P->template mutable_data<float>();
  const float* weights = (InputSize() > 2 ? Input(2).data<float>() : nullptr);

  if (label_prob_mode_) {
    CAFFE_ENFORCE_GE(T.dim(), 2);
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), D);
  } else {
    if (T.dim() == canonical_axis) {
      CAFFE_ENFORCE_EQ(T.numel(), N);
    } else {
      CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
      CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
    }
  }

  if (!losses_.defined()) {
    losses_ = caffe2::empty({N}, at::dtype<float>().device(CPU));
  } else if (losses_.numel() != N) {
    losses_.Resize(N);
  }

  softmax_utils::SoftmaxCPU<float>(
      N,
      D,
      !label_prob_mode_,
      X.data<float>(),
      Pdata,
      losses_.mutable_data<float>(),
      &context_);

  // Then compute cross entropy
  float loss_sum = 0.0;
  float weight_sum = 0.0;
  if (!label_prob_mode_) {
    const int* label_data = T.data<int>();

    for (int i = 0; i < N; ++i) {
      CAFFE_ENFORCE(
          label_data[i] < D && label_data[i] >= 0,
          "Label seems incorrect: label value larger than number of classes: ",
          label_data[i],
          " vs ",
          D);
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      float weight = weights ? weights[i] : 1.0;
      float l = -Pdata[i * D + label_data[i]] * weight;
      loss_sum += l;
      weight_sum += weight;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    math::Exp(N * D, Pdata, Pdata, &context_);
  } else {
    const float* label_data = T.data<float>();

    for (int i = 0; i < N; ++i) {
      float l = 0.0;
      float total_prob = 0.0;
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      float weight = weights ? weights[i] : 1.0;
      for (int j = 0; j < D; ++j) {
        CAFFE_ENFORCE(
            label_data[i * D + j] >= 0,
            "Label prob seems incorrect: label prob value must be nonnegative:",
            " ",
            label_data[i * D + j]);
        l += -log(std::max(Pdata[i * D + j], 1e-20f)) * label_data[i * D + j] *
            weight;
        total_prob += label_data[i * D + j];
      }
      loss_sum += l;
      CAFFE_ENFORCE(
          std::abs(total_prob - 1.) < 1e-5f,
          "Label prob seems incorrect: label prob values do not sum to 1.0: ",
          total_prob,
          " vs 1.0 (+/- 1e-5)");
      weight_sum += weight;
    }
  }

  auto* avg_loss =
      Output(1, vector<int64_t>(), at::dtype<float>()); // Average loss

  float* avg_loss_data = avg_loss->template mutable_data<float>();
  if (weight_sum != 0.0) {
    if (average_by_batch_size_) {
      avg_loss_data[0] = loss_sum * scale_ / N;
    } else {
      avg_loss_data[0] = loss_sum * scale_ / weight_sum;
    }
  } else {
    avg_loss_data[0] = 0.0;
  }
  return true;
}

template <>
bool SoftmaxWithLossGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0); // Logits
  auto& T = Input(1); // Labels / targets
  // Input(2) is weights if given
  auto& P = Input(InputSize() - 2); // Probabilities from softmax
  auto& d_avg_loss = Input(InputSize() - 1); // Gradient w.r.t. avg loss

  const float* weights = (InputSize() > 4 ? Input(2).data<float>() : nullptr);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  float avg_denominator;

  if (label_prob_mode_) {
    CAFFE_ENFORCE_GE(T.dim(), 2);
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), D);
  } else {
    if (T.dim() == canonical_axis) {
      CAFFE_ENFORCE_EQ(T.numel(), N);
    } else {
      CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
      CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
    }
  }

  const float* Pdata = P.data<float>();
  float* dX_data = dX->template mutable_data<float>();

  // Copy softmax probabilities into dX. All but the neuron
  // corresponding to the correct label has gradient equaling e(x_j)
  // which is the probability under softmax.
  context_.CopyFromCPU<float>(P.numel(), Pdata, dX_data);

  // Compute gradient for the matching labels.
  float total_weight = 0.0f;
  if (!label_prob_mode_) {
    const int* label_data = T.data<int>();

    if (weights) {
      for (int i = 0; i < N; ++i) {
        int idx = i * D + label_data[i];
        float weight = weights[i];
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        dX_data[idx] = Pdata[idx] - 1.0;
        for (int d = 0; d < D; d++) {
          int k = i * D + d;
          dX_data[k] *= weight;
        }

        total_weight += weight;
      }
    } else {
      for (int i = 0; i < N; ++i) {
        int idx = i * D + label_data[i];
        dX_data[idx] = Pdata[idx] - 1.0f;
      }
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      total_weight = N;
    }
  } else {
    const float* label_data = T.data<float>();

    if (weights) {
      for (int i = 0; i < N; ++i) {
        float weight = weights[i];
        for (int j = 0; j < D; ++j) {
          int idx = i * D + j;
          dX_data[idx] = (Pdata[idx] - label_data[idx]) * weight;
        }
        total_weight += weight;
      }
    } else {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
          int idx = i * D + j;
          dX_data[idx] = Pdata[idx] - label_data[idx];
        }
      }
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      total_weight = N;
    }
  }

  // Scale by d_avg_loss / N
  if (total_weight > 0) {
    if (average_by_batch_size_) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      avg_denominator = N;
    } else {
      avg_denominator = total_weight;
    }
    math::Scale<float, float, CPUContext>(
        dX->numel(),
        scale_ / avg_denominator * d_avg_loss.data<float>()[0],
        dX->data<float>(),
        dX_data,
        &context_);
  }
  return true;
}

namespace {
class GetSoftmaxWithLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> blob_names{
        {I(0), I(1), O(0), GO(1)},
    };

    // Add weight blob, if given
    if (def_.input_size() == 3) {
      blob_names.emplace(blob_names.begin() + 2, I(2));
    }
    return SingleGradientDef(
        "SoftmaxWithLossGradient", "", blob_names, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(SoftmaxWithLoss, GetSoftmaxWithLossGradient);
} // namespace
} // namespace caffe2

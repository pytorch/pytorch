#include "caffe2/operators/random_softmax_op.h"
#include <cfloat>
#include <limits>
#include "Eigen/Core"
#include "caffe2/operators/softmax_shared.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

// For each training example, sample K classes:
//    1. keep all classes where label is positive, and
//    2. randomly sample K - num_positive_labels
//      out of D - num_positive_labels negative label classes
// based on Knuth selection sampling technique
// (Algorithm S, The Art of Computer Programming 3.4.2)
// Input:
//    labels, size N * D,
//    num_postive_labels, size N * 1,
//      which stores num positive labels for each training example
// Output:
//    samples, size N * K, which stores classes that are sampled
void RandomSoftmaxSampling(
    const int N, // batch size
    const int D, // num all classes
    const int K, // num classes to sample
    const float* labels,
    const int* num_positive_labels, // num positive labels
    int* samples,
    CPUContext& context) {
  for (int i = 0; i < N; ++i) {
    int offset = i * D;

    int t = 0; // total negative labels dealt with
    int m = 0; // number of negative labels items selected so far
    int j = 0; // number of labels visited
    int k = num_positive_labels[i]; // number of positive labels
    int k0 = 0; // num positive labels selected so far
    float u = 0.0f;
    while (m < K - k) {
      // classes which have positive labels are always sampled
      if (labels[offset + j] >= 0.5f) {
        samples[i * K + k0] = j;
        k0++;
      } else {
        // call a uniform(0,1) random number generator
        math::RandUniform<float, CPUContext>(1, 0.0f, 1.0f, &u, &context);
        // pool size: D - k (total num negatives), sample size: K - k
        if ((D - k - t) * u < K - k - m) {
          samples[i * K + k + m] = j;
          m++;
        }
        t++;
      }
      j++;
    }

    // after K - k negative label classes has been sampled,
    // continue scanning the row to select the rest of positive label classes
    while (j < D && k0 < k) {
      if (labels[offset + j] >= 0.5f) {
        samples[i * K + k0] = j;
        k0++;
      }
      j++;
    }
  }
}

void CountPositiveLabels(
    const int N,
    const int D, // num all classes
    const float* labels,
    int* num_positive_labels) {
  for (int i = 0; i < N; ++i) {
    auto offset = i * D;
    int k = 0;
    for (int j = 0; j < D; ++j) {
      if (labels[offset + j] >= 0.5) {
        k++;
      }
    }
    num_positive_labels[i] = k;
  }
}

void SampleInput(
    const int N,
    const int D,
    const int K,
    const int* samples,
    const float* in,
    float* out) {
  for (int i = 0; i < N * K; ++i) {
    int row = i / K;
    out[i] = in[row * D + samples[i]];
  }
}

void SampleOutput(
    const int N,
    const int D,
    const int K,
    const int* samples,
    const float* in,
    float* out) {
  for (int i = 0; i < N * K; ++i) {
    int row = i / K;
    out[row * D + samples[i]] = in[i];
  }
}

void CrossEntropyLossCPU(
    const int N,
    const int D,
    const float* P_data,
    const float* label_data,
    float* avg_loss) {
  CAFFE_ENFORCE_GT(N, 0, "number of inputs cannot be zero");
  *avg_loss = -(ConstEigenArrayMap<float>(label_data, D, N) *
                ConstEigenArrayMap<float>(P_data, D, N).cwiseMax(FLT_MIN).log())
                   .sum();
  avg_loss[0] /= N;
}

// Implementation for the CPU context.
template <>
bool RandomSoftmaxOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0); // Logits
  auto& T = Input(1); // Labels/Targets

  CAFFE_ENFORCE(T.sizes() == X.sizes());

  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);
  const int K = num_sampled_;

  auto* sampled_P =
      Output(0, vector<int64_t>{N, K}, at::dtype<float>()); // sampled_softmax
  auto* sampled_T =
      Output(1, vector<int64_t>{N, K}, at::dtype<float>()); // sampled_labels
  auto* avg_loss =
      Output(2, vector<int64_t>(), at::dtype<float>()); // Average loss
  auto* samples = Output(
      3,
      vector<int64_t>{N, K},
      at::dtype<int>()); // size: (N, K), stored sampled classes. 0 <= elem < D
  auto* num_positive_labels =
      Output(4, vector<int64_t>{N}, at::dtype<int>()); // size: N

  float* sampled_P_data = sampled_P->template mutable_data<float>();
  float* sampled_label_data = sampled_T->template mutable_data<float>();
  float* avg_loss_data = avg_loss->template mutable_data<float>();
  int* samples_data = samples->template mutable_data<int>();

  if (N == 0) {
    return true;
  }

  CountPositiveLabels(
      N, D, T.data<float>(), num_positive_labels->mutable_data<int>());
  // If input_samples, size N * K, is provided
  if (InputSize() >= 3) {
    auto& input_samples = Input(2);
    context_.template CopyFromCPU<int>(
        N * K, input_samples.data<int>(), samples_data);
  } else {
    RandomSoftmaxSampling(
        N,
        D,
        K,
        T.data<float>(),
        num_positive_labels->data<int>(),
        samples_data,
        context_);
  }

  // sample input
  SampleInput(N, D, K, samples_data, X.data<float>(), sampled_P_data);
  SampleInput(N, D, K, samples_data, T.data<float>(), sampled_label_data);

  // First, get scales
  if (!scale_.defined()) {
    scale_ = caffe2::empty({N}, at::dtype<float>().device(CPU));
  } else if (scale_.numel() != N) {
    scale_.Resize(N);
  }

  if (!rowmax_.defined()) {
    rowmax_ = caffe2::empty({N}, at::dtype<float>().device(CPU));
  } else if (rowmax_.numel() != N) {
    rowmax_.Resize(N);
  }

  if (!sum_multiplier_.defined()) {
    sum_multiplier_ = caffe2::empty({K}, at::dtype<float>().device(CPU));
    math::Set<float, CPUContext>(
        K, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  } else if (sum_multiplier_.numel() != K) {
    sum_multiplier_.Resize(K);
    math::Set<float, CPUContext>(
        K, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }

  SoftmaxCPU(
      context_,
      N,
      K,
      sampled_P_data,
      sampled_P_data,
      scale_.mutable_data<float>(),
      sum_multiplier_.data<float>(),
      false,
      rowmax_.mutable_data<float>());

  // Then compute cross entropy
  CrossEntropyLossCPU(N, K, sampled_P_data, sampled_label_data, avg_loss_data);

  return true;
}

// Implementation for the CPU context.
template <>
bool RandomSoftmaxGradientOp<float, CPUContext>::RunOnDevice() {
  auto& sampled_P = Input(0); // sampled_labels
  auto& sampled_T = Input(1); // sampled_softmax
  auto& samples = Input(2); // samples
  auto& d_avg_loss = Input(3); // gradient of avg_loss
  auto& T = Input(4); // labels
  auto& num_positive_labels = Input(5); // num positive labels per row
  auto* dX = Output(0, T.sizes(), at::dtype<float>());

  const auto canonical_axis = sampled_P.canonical_axis_index(axis_);
  const int N = sampled_P.size_to_dim(canonical_axis);
  const int K = sampled_P.size_from_dim(canonical_axis);
  const int D = T.size_from_dim(canonical_axis);

  float* dX_data = dX->mutable_data<float>();
  if (N == 0) {
    return true;
  }

  if (!sampled_dX_.defined()) {
    sampled_dX_ = caffe2::empty({N * K}, at::dtype<float>().device(CPU));
  } else if (sampled_dX_.numel() != N * K) {
    sampled_dX_.Resize(N * K);
  }
  float* sampled_dX_data = sampled_dX_.mutable_data<float>();

  // Calculate gradient of input:
  // if idx is sampled, then gradient of input:
  //  = P[idx] * num_positive_labels[i] - label[idx], otherwise 0
  context_.CopyFromCPU<float>(
      sampled_P.numel(), sampled_P.data<float>(), sampled_dX_data);
  for (int i = 0; i < N; ++i) {
    auto offset = i * K;
    math::Scale<float, float, CPUContext>(
        K,
        num_positive_labels.data<int>()[i],
        sampled_dX_data + offset,
        sampled_dX_data + offset,
        &context_);
  }
  math::Sub<float, CPUContext>(
      sampled_P.numel(),
      sampled_dX_data,
      sampled_T.data<float>(),
      sampled_dX_data,
      &context_);
  math::Scale<float, float, CPUContext>(
      sampled_dX_.numel(),
      1.0 / N * d_avg_loss.data<float>()[0],
      sampled_dX_data,
      sampled_dX_data,
      &context_);

  math::Set<float, CPUContext>(T.size(), 0.0f, dX_data, &context_);
  SampleOutput(N, D, K, samples.data<int>(), sampled_dX_data, dX_data);
  return true;
}

REGISTER_CPU_OPERATOR(RandomSoftmax, RandomSoftmaxOp<float, CPUContext>);
REGISTER_CPU_GRADIENT_OPERATOR(
    RandomSoftmaxGradient,
    RandomSoftmaxGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(RandomSoftmax)
    .NumInputs(2, 3)
    .NumOutputs(5)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(

Applies the Random Softmax function to an n-dimensional input Tensor rescaling them so
that the elements of the n-dimensional output Tensor lie in the range (0,1).
The random softmax operator is typically the last layer in a classifier network,
as its output can be interpreted as confidence probabilities of an input belonging
to each class. The input is a 2-D tensor (Tensor) of size (batch_size x
input_feature_dimensions). The output tensor has the same shape and contains the
random softmax normalized values of the corresponding input. The random softmax function is
defined as follows:

$$random\_softmax(x_i) = \frac{\exp(x_i)}{\sum_{j \in C \cup \{i\}} \exp(x_j)}$$

The input does not need to explicitly be a 2D vector; rather, it will be coerced
into one. For an arbitrary n-dimensional tensor `X` in
$[a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}]$, where k is the `axis` provided,
then `X` will be coerced into a 2-dimensional tensor with dimensions
$[(a_0 * ... * a_{k-1}), (a_k * ... * a_{n-1})]$. For the default case where
`axis`=1, the `X` tensor will be coerced into a 2D tensor of dimensions
$[a_0, (a_1 * ... * a_{n-1})]$, where $a_0$ is often the batch size. In this
situation, we must have $a_0 = N$ and $a_1 * ... * a_{n-1} = D$. Each of these
dimensions must be matched correctly, or else the operator will throw errors.

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
        "RandomSoftmax",
        ["logits", "labels"],
        ["sampled_probs", "sampled_labels", "loss", "samples", "num_positive_labels"],
        num_sampled=K,
        engine=engine,
    )

workspace.FeedBlob("X", np.random.randn(1, 5).astype(np.float32))
print("input:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("random_softmax:", workspace.FetchBlob("Y"))

```

**Result**

```
input: ...
random_softmax: ...

```

</details>



)DOC")
    .Arg(
        "axis",
        "*(type: int; default: 1)* Axis of the inputs when coerced to 2D matrix.")
    .Arg(
        "num_samples",
        "*(type: int; default: 10)* Number of classes to sample")
    .Input(
        0,
        "logits",
        "*(type: Tensor`<float>`)* Input tensor that's coerced into a 2D matrix of size (NxD) as described above.")
    .Input(1, "labels", "*(type: Tensor`<float>`)* Ground truth label tensor.")
    .Input(
        2,
        "input_samples", // the hack to run unit test
        "Optional, mainly for unit test, samples tensor, shape (batch_size, num_samples) storing sampled classes. each elem is in [0, D)")
    .Output(
        0,
        "sampled_softmax",
        "*(type: Tensor`<float>`)* The random softmax output tensor with shape (batch_size, num_samples)")
    .Output(
        1,
        "sampled_labels",
        "*(type: Tensor`<float>`)* Labels of sampled classes. shape (batch_size, num_samples)")
    .Output(
        2,
        "avg_loss",
        "*(type: float)* Averaged cross-entropy loss on sampled indices.")
    .Output(
        3,
        "samples",
        "*(type: Tensor`<int>`)* samples tensor, shape (batch_size, num_samples) storing sampled classes. each elem is in [0, D)")
    .Output(
        4,
        "num_positive_labels",
        "*(type: Tensor`<int>`)* shape batch_size, num positive labels per training example")
    .InheritOnnxSchema();

// Input: sampled_labels, sampled_softmax, d_avg_loss, samples, labels,
// num_positive_labels. Output: d_logits
GRADIENT_OPERATOR_SCHEMA(RandomSoftmaxGradient).NumInputs(6).NumOutputs(1);

namespace {

class GetRandomSoftmaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        // sampled_softmax, sampled_labels, samples, d_avg_loss, labels,
        // num_positive_labels
        vector<string>{O(0), O(1), O(3), GO(2), I(1), O(4)},
        vector<string>{GI(0)});
  }
};

} // namespace
REGISTER_GRADIENT(RandomSoftmax, GetRandomSoftmaxGradient);
REGISTER_GRADIENT(RandomSoftmaxFp16, GetRandomSoftmaxGradient);

} // namespace caffe2

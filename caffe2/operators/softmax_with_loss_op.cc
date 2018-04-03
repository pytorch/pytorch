#include "softmax_with_loss_op.h"
#include "softmax_shared.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SoftmaxWithLoss, SoftmaxWithLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SoftmaxWithLossGradient,
    SoftmaxWithLossGradientOp<float, CPUContext>);

// Input: X (logits), T (labels); Output: P (probs), Y
OPERATOR_SCHEMA(SoftmaxWithLoss)
    .NumInputs(2, 3)
    .NumOutputs(2)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
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
Combined Softmax and Cross-Entropy loss operator.
The operator computes the softmax normalized values for each layer in the batch
of the given input, after which cross-entropy loss is computed. This operator is
numerically more stable than separate Softmax and CrossEntropy ops.
The inputs are a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions) and tensor of labels (ground truth).
Output is tensor with the probability for each label for each example (N x D)
and averaged loss (scalar).
Use parameter label_prob=1 to enable inputting labels as a probability
distribution.
Optional third input blob can be used to weight the samples for the loss.
)DOC")
    .Input(0, "logits", "Unscaled log probabilities")
    .Input(1, "labels", "Ground truth")
    .Input(
        2,
        "weight_tensor",
        "Optional blob to be used to weight the samples for the loss.")
    .Output(0, "softmax", "Tensor with softmax cross entropy loss")
    .Output(1, "loss", "Average loss");

// Input: X, T, P, dY; Output: dX
OPERATOR_SCHEMA(SoftmaxWithLossGradient).NumOutputs(1);

#define DONT_CARE (-1)

template <>
bool SoftmaxWithLossOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0); // Logits
  auto& T = Input(1); // Labels / targets
  auto* P = Output(0); // Probabilities from softmax
  auto* avg_loss = Output(1); // Average loss

  const auto canonical_axis = X.canonical_axis_index(axis_);
  int N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);
  P->ResizeLike(X);

  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CPUContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }

  float* Pdata = P->mutable_data<float>();
  const float* weights = (InputSize() > 2 ? Input(2).data<float>() : nullptr);

  if (label_prob_mode_) {
    CAFFE_ENFORCE_GE(T.ndim(), 2);
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), D);
  } else {
    if (T.ndim() == canonical_axis) {
      CAFFE_ENFORCE_EQ(T.size(), N);
    } else {
      CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
      CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
    }
  }

  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CPUContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }

  rowmax_.Resize(N);
  losses_.Resize(N);

  SoftmaxCPU(
      context_,
      N,
      D,
      X.data<float>(),
      Pdata,
      losses_.mutable_data<float>(),
      sum_multiplier_.data<float>(),
      !label_prob_mode_,
      rowmax_.mutable_data<float>());

  // Then compute cross entropy
  float loss_sum = 0.0;
  float weight_sum = 0.0;
  if (!label_prob_mode_) {
    const int* label_data = T.data<int>();
    const float* Xdata = X.data<float>();

    for (int i = 0; i < N; ++i) {
      CAFFE_ENFORCE(
          label_data[i] < D && label_data[i] >= 0,
          "Label seems incorrect: label value larger than number of classes: ",
          label_data[i],
          " vs ",
          D);
      float weight = weights ? weights[i] : 1.0;
      float l = -Pdata[i * D + label_data[i]] * weight;
      loss_sum += l;
      weight_sum += weight;
    }
    math::Exp(N * D, Pdata, Pdata, &context_);
  } else {
    const float* label_data = T.data<float>();

    for (int i = 0; i < N; ++i) {
      float l = 0.0;
      float total_prob = 0.0;
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

  avg_loss->Resize(vector<TIndex>());
  float* avg_loss_data = avg_loss->mutable_data<float>();
  if (weight_sum != 0.0) {
    avg_loss_data[0] = loss_sum * scale_ / weight_sum;
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
  auto* dX = Output(0);
  const float* weights = (InputSize() > 4 ? Input(2).data<float>() : nullptr);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  int N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);
  dX->ResizeLike(X);

  if (label_prob_mode_) {
    CAFFE_ENFORCE_GE(T.ndim(), 2);
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), D);
  } else {
    if (T.ndim() == canonical_axis) {
      CAFFE_ENFORCE_EQ(T.size(), N);
    } else {
      CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
      CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
    }
  }

  const float* Pdata = P.data<float>();
  float* dX_data = dX->mutable_data<float>();

  // Copy softmax probabilities into dX. All but the neuron
  // corresponding to the correct label has gradient equaling e(x_j)
  // which is the probability under softmax.
  context_.Copy<float, CPUContext, CPUContext>(P.size(), Pdata, dX_data);

  // Compute gradient for the matching labels.
  float total_weight = 0.0f;
  if (!label_prob_mode_) {
    const int* label_data = T.data<int>();

    if (weights) {
      for (int i = 0; i < N; ++i) {
        int idx = i * D + label_data[i];
        float weight = weights[i];
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
      total_weight = N;
    }
  }

  // Scale by d_avg_loss / N
  if (total_weight > 0) {
    math::Scale<float, CPUContext>(
        dX->size(),
        scale_ / total_weight * d_avg_loss.data<float>()[0],
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
}
} // namespace caffe2

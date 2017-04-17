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
          auto spatial_mode = helper.GetSingleArgument<int32_t>("spatial", 0);

          vector<TensorShape> out(2);

          auto logits = in[0]; // Tensor with Shape [batch_size, num_classes]
          auto labels = in[1]; // Tensor with same shape as logits

          auto batch_size = logits.dims().Get(0);
          auto num_classes = logits.dims().Get(1);

          if (!spatial_mode) {
            // Labels must only be 1D or 2D
            CAFFE_ENFORCE(labels.dims().size() <= 2);

            // Labels must have the same amount of elements as batch_size
            CAFFE_ENFORCE(batch_size == labels.dims().Get(0));

            out[0].set_data_type(logits.data_type());
            out[0].add_dims(batch_size);
            out[0].add_dims(num_classes);
          } else {
            DCHECK_EQ(logits.dims_size(), 4);
            DCHECK_EQ(labels.dims_size(), 3);
            out[0].set_data_type(logits.data_type());
            out[0].add_dims(batch_size);
            out[0].add_dims(num_classes);
            out[0].add_dims(in[0].dims(2));
            out[0].add_dims(in[0].dims(3));
          }

          // Output 2 is scalar shape, so no dims added
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
and averaged loss (scalar). Use parameter spatial=1 to enable spatial softmax.
Spatial softmax also supports special \"don't care\" label (-1) that is ignored
when computing the loss.
Use parameter label_prob=1 to enable inputting labels as a probability
distribution.  Currently does not handle spatial=1 case.
Optional third input blob can be used to weight the samples for the loss.
For the spatial version, weighting is by x,y position of the input.
)DOC")
    .Input(0, "logits", "Unscaled log probabilities")
    .Input(1, "labels", "Ground truth")
    .Input(
        2,
        "weight_tensor",
        "Optional blob to be used to weight the samples for the loss. With\
        spatial set, weighting is by x,y of the input")
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
  int N = X.dim32(0);
  int D = X.dim32(1);

  P->ResizeLike(X);

  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CPUContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }

  float* Pdata = P->mutable_data<float>();
  const float* weights = (InputSize() > 2 ? Input(2).data<float>() : nullptr);
  DCHECK(!(spatial_mode_ && label_prob_mode_));

  if (!spatial_mode_) {
    DCHECK_EQ(X.ndim(), 2);
    if (!label_prob_mode_) {
      DCHECK((T.ndim() == 1) || (T.ndim() == 2 && T.dim32(1) == 1));
    } else {
      DCHECK(T.ndim() == 2 && T.dim32(0) == N && T.dim32(1) == D);
    }
    DCHECK_EQ(T.dim32(0), N);

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
        CAFFE_ENFORCE(
            label_data[i] >= 0,
            "Label prob seems incorrect: label prob value must be nonnegative: ",
            label_data[i]);
        float l = 0.0;
        float total_prob = 0.0;
        float weight = weights ? weights[i] : 1.0;
        for (int j = 0; j < D; ++j) {
          l += -log(std::max(Pdata[i * D + j], 1e-20f)) *
              label_data[i * D + j] * weight;
          total_prob += label_data[i * D + j];
        }
        loss_sum += l;
        DCHECK(std::abs(total_prob - 1.) < 1e-5f);
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
  } else {
    // Spatial mode, compute softmax for each x, y location
    DCHECK_EQ(X.ndim(), 4);
    DCHECK_EQ(T.ndim(), 3);
    DCHECK_EQ(T.dim32(0), N);

    int H = X.dim32(2);
    int W = X.dim32(3);

    const float* Xdata = X.data<float>();

    for (int i = 0; i < N; ++i) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          // Subtract max on each cell for numerical reasons
          float max_val = (-1e20f);
          for (int c = 0; c < D; ++c) {
            // TODO optimize
            int idx = i * (H * W * D) + c * (H * W) + y * W + x;
            max_val = std::max(max_val, Xdata[idx]);
          }

          // Exponentiate
          float expsum = 0.0f;
          for (int c = 0; c < D; ++c) {
            int idx = i * (H * W * D) + c * (H * W) + y * W + x;
            float expx = exp(Xdata[idx] - max_val);
            Pdata[idx] = expx;
            expsum += expx;
          }

          // Normalize
          for (int c = 0; c < D; ++c) {
            int idx = i * (H * W * D) + c * (H * W) + y * W + x;
            Pdata[idx] /= expsum;
          }
        }
      }
    }

    // Compute the avg cross-entropy loss
    avg_loss->Resize(vector<TIndex>());
    float* avg_loss_data = avg_loss->mutable_data<float>();
    const int* label_data = T.data<int>();

    float sum_label_xent = 0.0f;
    float total_weight = 0.0;

    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        for (int i = 0; i < N; i++) {
          int label_idx = i * H * W + y * W + x;
          int label = label_data[label_idx];
          if (label != DONT_CARE) {
            CAFFE_ENFORCE(
                label < D && label >= 0,
                "Label seems incorrect: label value larger than number of classes: ",
                label_data[i],
                " vs ",
                D);
            int idx = i * (H * W * D) + label * (H * W) + y * W + x;
            float w = weights ? weights[label_idx] : 1.0;
            total_weight += w;
            sum_label_xent += -log(std::max(Pdata[idx], 1e-20f)) * w;
          }
        }
      }
    }
    if (total_weight != 0.0) {
      *avg_loss_data = sum_label_xent / total_weight;
    } else {
      *avg_loss_data = 0.0;
    }
  } // if spatial
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

  int N = X.dim32(0);
  int D = X.dim32(1);
  dX->ResizeLike(X);
  DCHECK_EQ(T.dim32(0), N);

  if (!spatial_mode_) {
    DCHECK_EQ(X.ndim(), 2);
    if (!label_prob_mode_) {
      DCHECK((T.ndim() == 1) || (T.ndim() == 2 && T.dim32(1) == 1));
    } else {
      DCHECK(T.ndim() == 2 && T.dim32(0) == N && T.dim32(1) == D);
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
  } else {
    // Spatial mode, compute softmax for each x, y location
    DCHECK_EQ(X.ndim(), 4);
    DCHECK_EQ(T.ndim(), 3);

    int H = X.dim32(2);
    int W = X.dim32(3);

    const float* Pdata = P.data<float>();
    float* dX_data = dX->mutable_data<float>();
    const int* label_data = T.data<int>();

    // Copy softmax probabilities into dX. All but the neuron
    // corresponding to the correct label has gradient equaling e(x_j)
    // which is the probability under softmax.
    context_.Copy<float, CPUContext, CPUContext>(P.size(), Pdata, dX_data);

    float total_weight = 0.0f;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        for (int i = 0; i < N; ++i) {
          int label_idx = i * H * W + y * W + x;
          int label = label_data[label_idx];

          if (label != DONT_CARE) {
            int idx = i * (H * W * D) + label * (H * W) + y * W + x;

            dX_data[idx] = (dX_data[idx] - 1.0);

            if (weights != nullptr) {
              float weight = weights[label_idx];
              for (int c = 0; c < D; ++c) {
                int k = i * (H * W * D) + c * (H * W) + y * W + x;
                dX_data[k] *= weight;
              }
              total_weight += weight;
            } else {
              total_weight += 1.0;
            }
          } else {
            // Set gradient to zero for coordinates where we have dont care
            for (int c = 0; c < D; ++c) {
              int idx = i * (H * W * D) + c * (H * W) + y * W + x;
              dX_data[idx] = 0;
            }
          }
        }
      }
    }

    if (total_weight > 0) {
      math::Scale<float, CPUContext>(
          dX->size(),
          scale_ / total_weight,
          dX->data<float>(),
          dX_data,
          &context_);
    }
    math::Scale<float, CPUContext>(
        dX->size(),
        d_avg_loss.data<float>(),
        dX->data<float>(),
        dX->mutable_data<float>(),
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

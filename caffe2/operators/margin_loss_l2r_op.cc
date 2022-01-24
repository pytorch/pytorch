#include "caffe2/operators/margin_loss_l2r_op.h"
#include <cmath>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

namespace {

#define PAIRWISE_DIFF(vec, N)                               \
  ((vec.matrix() * Eigen::MatrixXf::Ones(1, N) -            \
    Eigen::MatrixXf::Ones(N, 1) * vec.matrix().transpose()) \
       .array())
#define CWISE_GT(vec1, vec2) ((vec1) > (vec2))

#define CWISE_LT(vec1, vec2) ((vec1) < (vec2))

#define CWISE_SIGN(vec) \
  CWISE_GT((vec), 0).cast<float>() - CWISE_LT((vec), 0).cast<float>()
} // namespace

template <>
float SessionMarginLossOp<float, CPUContext>::SessionMarginLoss(
    int start_index,
    int end_index,
    const Tensor& pred,
    const Tensor& label,
    Tensor** dpred) {
  CAFFE_ENFORCE_LE(0.0, start_index);
  CAFFE_ENFORCE_GE(pred.numel(), start_index);
  const auto* pred_data = pred.template data<float>();
  const auto* label_data = label.template data<float>();
  int N = end_index - start_index + 1;
  ConstEigenVectorArrayMap<float> pred_vec(&pred_data[start_index], N);
  ConstEigenVectorArrayMap<float> label_vec(&label_data[start_index], N);
  auto* dpred_data = (*dpred)->template mutable_data<float>();
  EigenVectorArrayMap<float> dpred_vec(&dpred_data[start_index], N);
  dpred_vec = 0;

  ReinitializeTensor(&margin_diff_, {N * N}, at::dtype<float>().device(CPU));
  auto* margin_diff_data = margin_diff_.template mutable_data<float>();
  EigenArrayMap<float> margin_diff_mat(margin_diff_data, N, N);

  ReinitializeTensor(
      &label_relation_sign_, {N * N}, at::dtype<float>().device(CPU));
  auto* label_relation_sign_data =
      label_relation_sign_.template mutable_data<float>();
  EigenArrayMap<float> label_relation_sign_mat(label_relation_sign_data, N, N);

  // in case that all docs in a session have zero ratings, no op
  if (label_vec.abs().sum() < 1e-6) {
    return 0;
  }
  if (N <= 0) {
    return 0;
  }

  float weight = 1.0f / N;

  // define label relation, return N * N MATRIX, element (i, j) will be sign(label_i - label_i)
  label_relation_sign_mat = PAIRWISE_DIFF(label_vec, N).cwiseSign();
  margin_diff_mat =
      (margin_ - (label_relation_sign_mat * PAIRWISE_DIFF(pred_vec, N))) *
      label_relation_sign_mat.abs();
  float loss = 0.5f * weight *
      (margin_diff_mat * CWISE_GT(margin_diff_mat, 0).cast<float>()).sum();
  dpred_vec = -weight *
      ((CWISE_GT(margin_diff_mat, 0).cast<float>()) * label_relation_sign_mat)
          .rowwise()
          .sum();

  return loss;
}

template <>
bool SessionMarginLossOp<float, CPUContext>::RunOnDevice() {
  auto& pred = Input(PRED);
  auto& label = Input(LABEL);
  auto& sid = Input(SESSION_LENS);

  auto* dpred = Output(DPRED);

  const auto* session_lengths = sid.template data<int>();
  CAFFE_ENFORCE(pred.dim() == 1);
  CAFFE_ENFORCE(pred.numel() == label.numel());
  dpred->Resize(pred.numel());
  auto* loss = Output(LOSS, {sid.numel()}, at::dtype<float>());
  auto loss_vec = loss->template mutable_data<float>();
  int start_id = 0;
  for (int i = 0; i < sid.numel(); i++) {
    loss_vec[i] = SessionMarginLoss(
        start_id, session_lengths[i] + start_id - 1, pred, label, &dpred);
    start_id += session_lengths[i];
  }

  return true;
}

template <>
bool SessionMarginLossGradientOp<float, CPUContext>::RunOnDevice() {
  auto& pred = Input(PRED);
  auto& sids = Input(SESSION_LENS);
  auto& precomputed_dpred = Input(PRECOMPUTED_DPRED);
  auto& dLoss = Input(DLOSS);

  CAFFE_ENFORCE(pred.dim() == 1);
  CAFFE_ENFORCE(precomputed_dpred.dim() == 1);
  CAFFE_ENFORCE(precomputed_dpred.numel() > 0);
  CAFFE_ENFORCE(pred.numel() == precomputed_dpred.numel());

  const auto* session_lengths = sids.template data<int>();
  CAFFE_ENFORCE(dLoss.numel() == sids.numel());

  ConstEigenVectorArrayMap<float> precomputed_dpred_vec(
      precomputed_dpred.template data<float>(), precomputed_dpred.numel());
  auto* dpred = Output(DPRED, {precomputed_dpred.numel()}, at::dtype<float>());
  EigenVectorArrayMap<float> dpred_vec(
      dpred->template mutable_data<float>(), dpred->numel());
  auto multiplier = dLoss.template data<float>();
  int count = 0;
  for (int j = 0; j < sids.numel(); j++) {
    dpred_vec.segment(count, session_lengths[j]) = multiplier[j] *
        precomputed_dpred_vec.segment(count, session_lengths[j]);
    count += session_lengths[j];
  }
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(
    SessionMarginLoss,
    SessionMarginLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SessionMarginLossGradient,
    SessionMarginLossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SessionMarginLoss).NumInputs(3).NumOutputs(2).SetDoc(R"DOC(

This method optimizes the pairwise margin loss in a session with margin control.
If multiple sessions are in a batch, pairwise loss will only be computed in a session and the total loss will be the sum of pairwise loss from each session.
The exact loss function in a session is similar to https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss

)DOC");
OPERATOR_SCHEMA(SessionMarginLossGradient).NumInputs(4).NumOutputs(1);

class GetSessionMarginLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SessionMarginLossGradient",
        "",
        vector<string>{I(0), I(2), O(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(SessionMarginLoss, GetSessionMarginLossGradient);

} // namespace

} // namespace caffe2

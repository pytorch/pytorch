#include "caffe2/operators/listwise_l2r_op.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

namespace {

// Returns the indices that would sort an array. For example:
//   data = [3, 1, 2, 4]
//   return = [1, 2, 0, 3] (reverse = false)
//   return = [3, 0, 2, 1] (reverse = true)
template <typename TDATA, typename TIDX>
void arg_sort(const TDATA* data, TIDX* idx, const size_t N, bool reverse) {
  std::function<bool(size_t, size_t)> cmp_lambda;
  if (reverse) {
    cmp_lambda = [data](size_t i, size_t j) { return data[i] > data[j]; };
  } else {
    cmp_lambda = [data](size_t i, size_t j) { return data[i] < data[j]; };
  }
  size_t n = 0;
  std::generate(idx, idx + N, [&n] { return n++; });
  std::sort(idx, idx + N, cmp_lambda);
}

#define PAIRWISE_DIFF(vec, N)                               \
  ((vec.matrix() * Eigen::MatrixXf::Ones(1, N) -            \
    Eigen::MatrixXf::Ones(N, 1) * vec.matrix().transpose()) \
       .array())

#define CWISE_SIGM(vec) (1. / (1. + (-(vec)).exp()))

#define CWISE_GT(vec1, vec2) ((vec1) > (vec2))

#define CWISE_LT(vec1, vec2) ((vec1) < (vec2))

#define CWISE_SIGN(vec) (CWISE_GT((vec), 0).cast<float>() * 2. - 1.)

#define CWISE_LOG_SIGM(vec, huge) \
  (CWISE_GT((vec), (huge))        \
       .select(                   \
           0, CWISE_LT((vec), -(huge)).select(vec, CWISE_SIGM((vec)).log())))

} // namespace

template <>
void LambdaRankNdcgOp<float, CPUContext>::ResizeInvLogITensor(int size) {
  int old_size = inv_log_i_.numel();
  int new_size = std::max(old_size, 1);
  while (new_size < size) {
    new_size <<= 1;
  }
  if (new_size != old_size) {
    ReinitializeTensor(&inv_log_i_, {new_size}, at::dtype<float>().device(CPU));
    auto* data = inv_log_i_.template mutable_data<float>();
    EigenVectorArrayMap<float> vec(data, inv_log_i_.numel());
    const float log2f_ = std::log(2.f);
    vec = log2f_ *
        (Eigen::ArrayXf::LinSpaced(new_size, 2, 1 + new_size).log().inverse());
  }
  return;
}

template <>
void LambdaRankNdcgOp<float, CPUContext>::ComputeDiscounts(int* idx, int N) {
  ReinitializeTensor(&discount_, {N}, at::dtype<float>().device(CPU));
  auto* discount_data = discount_.template mutable_data<float>();
  auto* inv_log_i_data = inv_log_i_.template mutable_data<float>();
  for (int i = 0; i < N; i++) {
    discount_data[idx[i]] = inv_log_i_data[i];
  }
  return;
}

template <>
float LambdaRankNdcgOp<float, CPUContext>::LambdaRankNdcgSession(
    int start_index,
    int end_index,
    const Tensor& y,
    const Tensor& r,
    Tensor** dy) {
  CAFFE_ENFORCE(start_index >= 0);
  CAFFE_ENFORCE(start_index < y.numel());
  const auto* y_data = y.template data<float>();
  const auto* r_data = r.template data<float>();

  int N = end_index - start_index + 1;

  ConstEigenVectorArrayMap<float> y_vec(&y_data[start_index], N);
  ConstEigenVectorArrayMap<float> r_vec(&r_data[start_index], N);

  if (N <= 0) {
    return 0;
  }

  ReinitializeTensor(&ideal_idx_, {N}, at::dtype<int>().device(CPU));
  ReinitializeTensor(&rank_idx_, {N}, at::dtype<int>().device(CPU));
  auto* rank_idx_data = rank_idx_.template mutable_data<int>();
  auto* ideal_idx_data = ideal_idx_.template mutable_data<int>();

  // current ranked list is obtained by sorting by current score
  arg_sort(&y_data[start_index], rank_idx_data, N, true);
  // ideal ranked list is same as sorting by label
  arg_sort(&r_data[start_index], ideal_idx_data, N, true);

  auto* dy_data = (*dy)->template mutable_data<float>();
  EigenVectorArrayMap<float> dy_vec(&dy_data[start_index], N);
  float loss = 0;
  dy_vec = 0;
  // in case that all docs in a session have zero ratings, no op
  if (r_vec.abs().sum() < 1e-6) {
    return 0;
  }

  const double log2f_ = std::log(2.f);
  ReinitializeTensor(&gain_, {N}, at::dtype<float>().device(CPU));
  auto* gain_data = gain_.template mutable_data<float>();
  EigenVectorArrayMap<float> gain_vec(gain_data, gain_.numel());

  if (use_ndcg_as_loss_ && !use_exp_gain_) {
    gain_vec = r_vec;
  } else {
    // Gain vector = 2^rel = exp{rel * log(2)}
    gain_vec = (r_vec * log2f_).exp();
  }
  ResizeInvLogITensor(N);
  ComputeDiscounts(ideal_idx_data, N);
  auto* ideal_discount_data = discount_.template mutable_data<float>();
  EigenVectorArrayMap<float> ideal_discount_vec(
      ideal_discount_data, discount_.numel());
  // ideal dcg = \sum gain_i * ideal_discount_i
  double idcg = (gain_vec * ideal_discount_vec).sum();

  ComputeDiscounts(rank_idx_data, N);
  auto* discount_data = discount_.template mutable_data<float>();
  EigenVectorArrayMap<float> discount_vec(discount_data, discount_.numel());
  // similar to ideal but replace with actual discounts
  double dcg = (gain_vec * discount_vec).sum();

  ReinitializeTensor(&lambda_, {N * N}, at::dtype<float>().device(CPU));
  auto* lambda_data = lambda_.template mutable_data<float>();
  EigenArrayMap<float> lambda_mat(lambda_data, N, N);
  // computes lambda weight (i, j) = abs(gain_dff * discount_diff)
  lambda_mat =
      (PAIRWISE_DIFF(discount_vec, N) * PAIRWISE_DIFF(gain_vec, N)).abs();

  // dy_i =
  //    \sum_j lambda_{i, j} -sign(i > j) * sigm( -sign(i > j)*(yi - yj) )
  //                         |++ gradient of rank loss between i & j  ++|
  dy_vec =
      -(lambda_mat * CWISE_SIGN(PAIRWISE_DIFF(r_vec, N)) *
        CWISE_SIGM(
            -CWISE_SIGN(PAIRWISE_DIFF(r_vec, N)) * PAIRWISE_DIFF(y_vec, N)))
           .rowwise()
           .sum();
  if (use_ndcg_as_loss_) {
    // DCG loss function
    loss = (idcg - dcg);
  } else {
    loss = -(lambda_mat *
             CWISE_LOG_SIGM(
                 CWISE_SIGN(PAIRWISE_DIFF(r_vec, N)) * PAIRWISE_DIFF(y_vec, N),
                 100))
                .sum();
  }

  // if use_idcg_normalization_ is true, the loss function is normalized by idcg
  // (e.g. NDCG), else un-normalized loss function (e.g. DCG)
  // Note that normalization is mathematically correct if idcg is guaranteed to
  // be positive!
  if (use_idcg_normalization_) {
    dy_vec /= std::max(idcg, 1e-5);
    loss /= std::max(idcg, 1e-5);
  }
  return loss;
}

template <>
bool LambdaRankNdcgOp<float, CPUContext>::RunOnDevice() {
  auto& y = Input(PRED);
  auto& r = Input(REL);
  auto& sid = Input(SESSION_LENS);

  auto* dy = Output(DPRED);

  const auto* session_lengths = sid.template data<int>();
  CAFFE_ENFORCE(y.dim() == 1);
  CAFFE_ENFORCE(y.numel() == r.numel());
  dy->Resize(y.numel());
  auto* loss = Output(LOSS, {sid.numel()}, at::dtype<float>());
  auto loss_vec = loss->template mutable_data<float>();
  int start_id = 0;
  for (int i = 0; i < sid.numel(); i++) {
    loss_vec[i] = LambdaRankNdcgSession(
        start_id, session_lengths[i] + start_id - 1, y, r, &dy);
    start_id += session_lengths[i];
  }

  return true;
}

template <>
bool LambdaRankNdcgGradientOp<float, CPUContext>::RunOnDevice() {
  auto& y = Input(Y);
  auto& sids = Input(SESSION_LENS);
  auto& dy_cache = Input(DY_CACHE);
  auto& dLoss = Input(DLOSS);

  CAFFE_ENFORCE(y.dim() == 1);
  CAFFE_ENFORCE(dy_cache.dim() == 1);
  CAFFE_ENFORCE(dy_cache.numel() > 0);
  CAFFE_ENFORCE(y.numel() == dy_cache.numel());

  const auto* session_lengths = sids.template data<int>();
  CAFFE_ENFORCE(dLoss.numel() == sids.numel());

  ConstEigenVectorArrayMap<float> dy_cache_vec(
      dy_cache.template data<float>(), dy_cache.numel());
  auto* dy = Output(DY, {dy_cache.numel()}, at::dtype<float>());
  EigenVectorArrayMap<float> dy_vec(
      dy->template mutable_data<float>(), dy->numel());
  auto multiplier = dLoss.template data<float>();
  int count = 0;
  for (int j = 0; j < sids.numel(); j++) {
    dy_vec.segment(count, session_lengths[j]) =
        multiplier[j] * dy_cache_vec.segment(count, session_lengths[j]);
    count += session_lengths[j];
  }
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(LambdaRankNdcg, LambdaRankNdcgOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    LambdaRankNdcgGradient,
    LambdaRankNdcgGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LambdaRankNdcg).NumInputs(3).NumOutputs(2).SetDoc(R"DOC(
It implements the LambdaRank as appeared in Wu, Qiang, et al. "Adapting boosting
for information retrieval measures." Information Retrieval 13.3 (2010): 254-270.

This method heuristically optimizes the NDCG.
)DOC");
OPERATOR_SCHEMA(LambdaRankNdcgGradient).NumInputs(4).NumOutputs(1);

class GetLambdaRankNdcgGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "LambdaRankNdcgGradient",
        "",
        vector<string>{I(0), I(2), O(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(LambdaRankNdcg, GetLambdaRankNdcgGradient);

} // namespace

} // namespace caffe2

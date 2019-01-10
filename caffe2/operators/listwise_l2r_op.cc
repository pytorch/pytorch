#include "caffe2/operators/listwise_l2r_op.h"

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
  int old_size = inv_log_i_.size();
  int new_size = std::max(old_size, 1);
  while (new_size < size) {
    new_size <<= 1;
  }
  if (new_size != old_size) {
    inv_log_i_.Resize(new_size);
    auto* data = inv_log_i_.template mutable_data<float>();
    EigenVectorArrayMap<float> vec(data, inv_log_i_.size());
    const float log2f_ = std::log(2.f);
    vec = log2f_ *
        (Eigen::ArrayXf::LinSpaced(new_size, 2, 1 + new_size).log().inverse());
  }
  return;
}

template <>
void LambdaRankNdcgOp<float, CPUContext>::ComputeDiscounts(int* idx, int N) {
  discount_.Resize(N);
  auto* discount_data = discount_.template mutable_data<float>();
  auto* inv_log_i_data = inv_log_i_.template mutable_data<float>();
  for (int i = 0; i < N; i++) {
    discount_data[idx[i]] = inv_log_i_data[i];
  }
  return;
}

template <>
bool LambdaRankNdcgOp<float, CPUContext>::RunOnDevice() {
  auto& y = Input(PRED);
  auto& r = Input(REL);
  auto* loss = Output(LOSS);
  auto* dy = Output(DPRED);

  const auto* y_data = y.template data<float>();
  const auto* r_data = r.template data<float>();
  ConstEigenVectorArrayMap<float> y_vec(y_data, y.size());
  ConstEigenVectorArrayMap<float> r_vec(r_data, r.size());
  CAFFE_ENFORCE(y.ndim() == 1);
  CAFFE_ENFORCE(y.size() == r.size());

  int N = y.size();
  ideal_idx_.Resize(N);
  rank_idx_.Resize(N);
  auto* rank_idx_data = rank_idx_.template mutable_data<int>();
  auto* ideal_idx_data = ideal_idx_.template mutable_data<int>();
  arg_sort(y_data, rank_idx_data, N, true);
  arg_sort(r_data, ideal_idx_data, N, true);

  const double log2f_ = std::log(2.f);
  gain_.Resize(N);
  auto* gain_data = gain_.template mutable_data<float>();
  EigenVectorArrayMap<float> gain_vec(gain_data, gain_.size());
  gain_vec = (r_vec * log2f_).exp();

  ResizeInvLogITensor(N);
  ComputeDiscounts(ideal_idx_data, N);
  auto* ideal_discount_data = discount_.template mutable_data<float>();
  EigenVectorArrayMap<float> ideal_discount_vec(
      ideal_discount_data, discount_.size());
  double idcg = (gain_vec * ideal_discount_vec).sum();
  // in case that all docs in a session have zero ratings, idcg will be zero.
  // For that case we will not normalize dcg.
  if (idcg < 1e-5) {
    idcg = 1.0;
  }

  ComputeDiscounts(rank_idx_data, N);
  auto* discount_data = discount_.template mutable_data<float>();
  EigenVectorArrayMap<float> discount_vec(discount_data, discount_.size());

  lambda_.Resize(N * N);
  auto* lambda_data = lambda_.template mutable_data<float>();
  EigenArrayMap<float> lambda_mat(lambda_data, N, N);
  // computes lambda weight (i, j) = abs(gain_dff * discount_diff)
  lambda_mat =
      (PAIRWISE_DIFF(discount_vec, N) * PAIRWISE_DIFF(gain_vec, N)).abs();

  loss->Resize(1);
  dy->Resize(N);
  auto* loss_data = loss->template mutable_data<float>();
  auto* dy_data = dy->template mutable_data<float>();
  EigenVectorArrayMap<float> dy_vec(dy_data, dy->size());
  // dy_i =
  //    \sum_j lambda_{i, j} -sign(i > j) * sigm( -sign(i > j)*(yi - yj) )
  //                         |++ gradient of rank loss between i & j  ++|
  dy_vec =
      -(lambda_mat * CWISE_SIGN(PAIRWISE_DIFF(r_vec, N)) *
        CWISE_SIGM(
            -CWISE_SIGN(PAIRWISE_DIFF(r_vec, N)) * PAIRWISE_DIFF(y_vec, N)))
           .rowwise()
           .sum() /
      idcg;
  // loss = \sum_{i, j} lambda_{i, j} rank_loss(i, j)
  *loss_data =
      -(lambda_mat *
        CWISE_LOG_SIGM(
            CWISE_SIGN(PAIRWISE_DIFF(r_vec, N)) * PAIRWISE_DIFF(y_vec, N), 100))
           .sum() /
      idcg;
  return true;
}

template <>
bool LambdaRankNdcgGradientOp<float, CPUContext>::RunOnDevice() {
  auto& y = Input(Y);
  auto& dy_cache = Input(DY_CACHE);
  auto& dLoss = Input(DLOSS);
  auto* dy = Output(DY);
  CAFFE_ENFORCE(y.ndim() == 1);
  CAFFE_ENFORCE(dy_cache.ndim() == 1);
  CAFFE_ENFORCE(dy_cache.size() > 0);
  CAFFE_ENFORCE(y.size() == dy_cache.size());
  CAFFE_ENFORCE(dLoss.size() == 1);

  ConstEigenVectorArrayMap<float> dy_cache_vec(
      dy_cache.template data<float>(), dy_cache.size());
  dy->Resize(dy_cache.size());
  EigenVectorArrayMap<float> dy_vec(
      dy->template mutable_data<float>(), dy->size());
  float multiplier = dLoss.template data<float>()[0];
  dy_vec = multiplier * dy_cache_vec;
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(LambdaRankNdcg, LambdaRankNdcgOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    LambdaRankNdcgGradient,
    LambdaRankNdcgGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LambdaRankNdcg).NumInputs(2).NumOutputs(2).SetDoc(R"DOC(
It implements the LambdaRank as appeared in Wu, Qiang, et al. "Adapting boosting
for information retrieval measures." Information Retrieval 13.3 (2010): 254-270.

This method heuristically optimizes the NDCG.
)DOC");
OPERATOR_SCHEMA(LambdaRankNdcgGradient).NumInputs(3).NumOutputs(1);

class GetLambdaRankNdcgGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "LambdaRankNdcgGradient",
        "",
        vector<string>{I(0), O(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(LambdaRankNdcg, GetLambdaRankNdcgGradient);

} // namespace

} // namespace caffe2

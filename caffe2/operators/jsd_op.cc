#include "caffe2/operators/jsd_op.h"

namespace caffe2 {

namespace {

static constexpr float kLOG_THRESHOLD() {
  return 1e-20;
}

inline float logit(float p) {
  // it computes log(p / (1-p))
  // to avoid numeric issue, hard code p log(p) when p approaches 0
  float x = std::min(std::max(p, kLOG_THRESHOLD()), 1 - kLOG_THRESHOLD());
  return -log(1. / x - 1.);
}

inline float entropy(float p) {
  if (p < kLOG_THRESHOLD() || 1 - p < kLOG_THRESHOLD()) {
    return 0.;
  } else {
    float q = 1 - p;
    return -p * log(p) - q * log(q);
  }
}
} // namespace

template <>
bool BernoulliJSDOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0); // predicted probabilities
  auto& T = Input(1); // target probabilities
  int N = X.numel();
  CAFFE_ENFORCE_EQ(T.numel(), N);
  auto* L = Output(0, X.sizes(), at::dtype<float>()); // JSD loss output
  auto* x_data = X.data<float>();
  auto* t_data = T.data<float>();
  auto* l_data = L->template mutable_data<float>();
  for (int i = 0; i < N; i++) {
    auto p_mdl = x_data[i];
    auto p_emp = t_data[i];
    auto p_avg = (p_mdl + p_emp) / 2.;
    auto jsd = entropy(p_avg) - (entropy(p_mdl) + entropy(p_emp)) / 2.;
    l_data[i] = jsd;
  }
  return true;
}

template <>
bool BernoulliJSDGradientOp<float, CPUContext>::RunOnDevice() {
  auto& go = Input(0);
  auto& X = Input(1);
  auto& T = Input(2);

  int N = X.numel();
  auto* gi = Output(0, X.sizes(), at::dtype<float>());
  auto* go_data = go.data<float>();
  auto* x_data = X.data<float>();
  auto* t_data = T.data<float>();
  auto* gi_data = gi->template mutable_data<float>();
  for (int i = 0; i < N; i++) {
    auto p_mdl = x_data[i];
    auto p_emp = t_data[i];
    auto p_avg = (p_mdl + p_emp) / 2.;
    auto g_jsd = (logit(p_mdl) - logit(p_avg)) / 2.;
    gi_data[i] = go_data[i] * g_jsd;
  }
  return true;
}
REGISTER_CPU_OPERATOR(BernoulliJSD, BernoulliJSDOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    BernoulliJSDGradient,
    BernoulliJSDGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(BernoulliJSD)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Computes the Jensen-Shannon divergence (JSD) between two Bernoulli distributions
where each is parametrized by a single probability.
)DOC")
    .Input(0, "X", "array of probabilities for prediction")
    .Input(0, "T", "array of probabilities for target")
    .Output(0, "L", "array of JSD losses");
OPERATOR_SCHEMA(BernoulliJSDGradient).NumInputs(3).NumOutputs(1);

class GetBernoulliJSDGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "BernoulliJSDGradient",
        "",
        vector<string>{GO(0), I(0), I(1)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(BernoulliJSD, GetBernoulliJSDGradient);

} // namespace caffe2

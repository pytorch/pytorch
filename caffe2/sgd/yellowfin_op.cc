#include "caffe2/sgd/yellowfin_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(YellowFin, YellowFinOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(YellowFin)
    .NumInputs(10)
    .NumOutputs(8)
    .AllowInplace(
        {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}})
    .SetDoc(R"DOC(

Computes the YellowFin update (https://arxiv.org/abs/1706.03471) and performs
momentum SGD optimization step. lr and mu are not being shared between
parameters. curv_win, g_avg, g2_avg and scalars_memory are just auxiliary
memory for computing moving averages (see the publication). Takes arguments
beta: coefficient for moving averages,
curv_win_width: timeframe when average squared gradient is being stored,
epsilon: for numerical purposes,
nesterov and zero_debias for debias of moving average.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Momentum")
    .Input(2, "lr", "Learning rate")
    .Input(3, "mu", "Momentum coefficient")
    .Input(4, "curv_win", "Memory for latest curvature ranges")
    .Input(5, "g_avg", "Moving average of gradient")
    .Input(6, "g2_avg", "Moving average of squared gradient")
    .Input(7, "scalars_memory", "Memory for stateful scalars")
    .Input(8, "grad", "Gradient computed")
    .Input(9, "iter", "Iteration number")
    .Output(0, "output_param", "Parameters to be updated")
    .Output(1, "output_moment", "Momentum")
    .Output(2, "output_lr", "Output learning rate")
    .Output(3, "output_mu", "Output momentum coefficient")
    .Output(4, "output_curv_win", "Output memory for latest curvature ranges")
    .Output(5, "output_g_avg", "Output moving average of gradient")
    .Output(6, "output_g2_avg", "Output moving average of squared gradient")
    .Output(7, "output_scalars_memory", "Output memory for stateful scalars")
    .Arg("beta", "Default 0.999")
    .Arg("curv_win_width", "Default 20")
    .Arg("epsilon", "Default 1e-6")
    .Arg("nesterov", "Default false")
    .Arg("zero_debias", "Default true");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(YellowFin);

#define CAFFE2_YELLOWFIN_GETLRMU(T)                                         \
  template <>                                                               \
  void YellowFinOp<T, CPUContext>::GetLrMu() {                              \
    const T curv_ratio = std::sqrt(*g_norm2_max_deb_ / *g_norm2_min_deb_);  \
    const T mu_limit = (curv_ratio - 1.0f) / (curv_ratio + 1.0f);           \
    const T pre_p = *distance_deb_ * *g_norm2_min_deb_;                     \
    const T p = (pre_p * pre_p) / (2.0f * *variance_);                      \
    const T w3 = (-std::sqrt(p * p + 4.0f / 27.0f * p * p * p) - p) / 2.0f; \
    const T w3_sign = w3 > 0.0f ? 1.0f : -1.0f;                             \
    const T w = w3_sign * std::pow(std::abs(w3), 1.0f / 3.0f);              \
    const T y = w - p / 3.0f / w;                                           \
    const T root = y + 1.0f;                                                \
    *mu_ = std::max(root * root, mu_limit * mu_limit);                      \
    *lr_ = std::pow(1.0f - std::sqrt(*mu_), 2) / *g_norm2_min_deb_;         \
    MovingAverage(1, mu_, mu_avg_, mu_avg_out_, mu_deb_);                   \
    MovingAverage(1, lr_, lr_avg_, lr_avg_out_, lr_deb_);                   \
  }

CAFFE2_YELLOWFIN_GETLRMU(float)
#undef CAFFE2_YELLOWFIN_GETLRMU

// Usually moment_ == moment_out_ && param_ == param_out_
#define CAFFE2_YELLOWFIN_MOMENTUMSGDUPDATE(T)                                  \
  template <>                                                                  \
  void YellowFinOp<T, CPUContext>::MomentumSgdUpdate() {                       \
    const T mu = *mu_avg_out_;                                                 \
    const T lr = *lr_avg_out_;                                                 \
    if (!nesterov_) {                                                          \
      for (int i = 0; i < D_; ++i) {                                           \
        moment_out_[i] = mu * moment_[i] + lr * grad_[i];                      \
        param_out_[i] = param_[i] - moment_out_[i];                            \
      }                                                                        \
    } else {                                                                   \
      for (int i = 0; i < D_; ++i) {                                           \
        const T moment_i = moment_[i];                                         \
        moment_out_[i] = mu * moment_i + lr * grad_[i];                        \
        param_out_[i] = param_[i] - (1 + mu) * moment_out_[i] + mu * moment_i; \
      }                                                                        \
    }                                                                          \
  }

CAFFE2_YELLOWFIN_MOMENTUMSGDUPDATE(float)
#undef CAFFE2_YELLOWFIN_MOMENTUMSGDUPDATE

} // caffe2

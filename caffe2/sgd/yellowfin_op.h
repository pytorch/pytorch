// YellowFin: An automatic tuner for momentum SGD
// (https://arxiv.org/abs/1706.03471)
// The YellowFinOp tunes learning rate and momentum and performs momentum SGD
// steps. The learning rate and momentum are separate for any matrix of
// parameters.

#pragma once

#include <cmath>
#include <cstring>
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class YellowFinOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  YellowFinOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        curv_win_width_(
            this->template GetSingleArgument<int>("curv_win_width", 20)),
        nesterov_(this->template GetSingleArgument<int>("nesterov", false)),
        zero_debias_(
            this->template GetSingleArgument<bool>("zero_debias", true)),
        epsilon_(this->template GetSingleArgument<T>("epsilon", 1e-6f)),
        beta_(this->template GetSingleArgument<T>("beta", 0.999f)) {}

 protected:
  // GetLrMu and MomentumSgdUpdate have different implementations for GPU and
  // CPU. All other methods are generic.
  void GetLrMu();
  void MomentumSgdUpdate();

  void AfterApply() {
    // g
    MovingAverage(D_, grad_, g_avg_, g_avg_out_, g_deb_);
    // g2
    math::Mul(D_, grad_, grad_, aux_vector_, &context_);
    MovingAverage(D_, aux_vector_, g2_avg_, g2_avg_out_, g2_deb_);
    // g_norm2
    math::Dot(D_, grad_, grad_, g_norm2_, &context_);
    math::Maximum(1, epsilon_, g_norm2_, g_norm2_, &context_);
    MovingAverage(1, g_norm2_, g_norm2_avg_, g_norm2_avg_out_, g_norm2_deb_);
    // g_norm
    math::Sqrt(1, g_norm2_, g_norm_, &context_);
    MovingAverage(1, g_norm_, g_norm_avg_, g_norm_avg_out_, g_norm_deb_);
    math::Maximum(1, epsilon_, g_norm_deb_, g_norm_deb_, &context_);
    // Curvature range: g_norm2_min, g_norm2_max
    math::CopyVector(curv_win_width_, curv_win_, curv_win_out_, &context_);
    T* curv_win_cell = curv_win_out_ + (iter_ - 1) % curv_win_width_;
    math::Log(1, g_norm2_, curv_win_cell, &context_);
    int valid_end = std::min(curv_win_width_, iter_);
    math::ReduceMin(
        valid_end, curv_win_out_, g_norm2_min_, &scratch_tensor_, &context_);
    math::ReduceMax(
        valid_end, curv_win_out_, g_norm2_max_, &scratch_tensor_, &context_);
    MovingAverage(
        1,
        g_norm2_min_,
        g_norm2_min_avg_,
        g_norm2_min_avg_out_,
        g_norm2_min_deb_);
    MovingAverage(
        1,
        g_norm2_max_,
        g_norm2_max_avg_,
        g_norm2_max_avg_out_,
        g_norm2_max_deb_);
    math::Exp(1, g_norm2_min_deb_, g_norm2_min_deb_, &context_);
    math::Exp(1, g_norm2_max_deb_, g_norm2_max_deb_, &context_);
    math::Maximum(1, epsilon_, g_norm2_min_deb_, g_norm2_min_deb_, &context_);
    math::Maximum(1, epsilon_, g_norm2_max_deb_, g_norm2_max_deb_, &context_);
    // Gradient variance
    math::Dot(D_, g_deb_, g_deb_, aux_scalar_, &context_);

    math::Sub(1, g_norm2_deb_, aux_scalar_, variance_, &context_);
    math::Maximum(1, epsilon_, variance_, variance_, &context_);
    // Distance to opt
    math::Div(1, g_norm_avg_out_, g_norm2_avg_out_, distance_, &context_);
    MovingAverage(
        1, distance_, distance_avg_, distance_avg_out_, distance_deb_);
    if (iter_ > 1) {
      GetLrMu();
    }
  }

  void MovingAverage(
      const int N,
      const T* elt,
      const T* avg,
      T* new_avg,
      T* debias_avg) {
    const T one = 1;
    math::Scale(N, beta_, avg, new_avg, &context_);
    math::Axpy(N, one - beta_, elt, new_avg, &context_);
    math::Scale(N, debias_factor_, new_avg, debias_avg, &context_);
  }

  T ZeroDebiasFactor() {
    if (zero_debias_) {
      const T one = 1;
      return one / (one - std::pow(beta_, iter_));
    } else {
      return 1;
    }
  }

 public:
  bool RunOnDevice() override {
// Iter live on the CPU

#define CAFFE2_YF_READ_INPUT(INPUT_NAME, VAR_NAME)   \
  const auto& VAR_NAME##_tensor = Input(INPUT_NAME); \
  VAR_NAME##_ = VAR_NAME##_tensor.template data<T>();

CAFFE2_YF_READ_INPUT(PARAM, param)
CAFFE2_YF_READ_INPUT(MOMENT, moment)
CAFFE2_YF_READ_INPUT(LR_AVG, lr_avg)
CAFFE2_YF_READ_INPUT(MU_AVG, mu_avg)
CAFFE2_YF_READ_INPUT(CURV_WIN, curv_win)
CAFFE2_YF_READ_INPUT(G_AVG, g_avg)
CAFFE2_YF_READ_INPUT(G2_AVG, g2_avg)
CAFFE2_YF_READ_INPUT(SCALARS_MEMORY, scalars_memory)
CAFFE2_YF_READ_INPUT(GRAD, grad)
#undef CAFFE2_YF_READ_OUTPUT

CAFFE_ENFORCE(OperatorBase::InputIsTensorType(ITER, CPU));
CAFFE_ENFORCE_EQ(lr_avg_tensor.numel(), 1);
CAFFE_ENFORCE_EQ(mu_avg_tensor.numel(), 1);
CAFFE_ENFORCE_EQ(param_tensor.dim(), moment_tensor.dim());
CAFFE_ENFORCE_EQ(param_tensor.dim(), g_avg_tensor.dim());
CAFFE_ENFORCE_EQ(param_tensor.dim(), g2_avg_tensor.dim());
CAFFE_ENFORCE_EQ(param_tensor.dim(), grad_tensor.dim());
for (const auto i : c10::irange(param_tensor.dim())) {
  CAFFE_ENFORCE_EQ(param_tensor.dim32(i), moment_tensor.dim32(i));
  CAFFE_ENFORCE_EQ(param_tensor.dim32(i), g_avg_tensor.dim32(i));
  CAFFE_ENFORCE_EQ(param_tensor.dim32(i), g2_avg_tensor.dim32(i));
  CAFFE_ENFORCE_EQ(param_tensor.dim32(i), grad_tensor.dim32(i));
}

    iter_ = OperatorBase::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];

    D_ = param_tensor.numel();

    // Input data - persistent memory for internal scalars
    // Note: Memory for these scalars is being allocated during initialization
    //       of the network. If you want to add / remove a scalar, make a
    //       suitable change of memory size in the initialization.
    const T* memory_it = scalars_memory_ - 1;
    g_norm_avg_ = ++memory_it;
    g_norm2_avg_ = ++memory_it;
    g_norm2_min_avg_ = ++memory_it;
    g_norm2_max_avg_ = ++memory_it;
    distance_avg_ = ++memory_it;

// Output data

#define CAFFE2_YF_READ_OUTPUT(OUTPUT_NAME, VAR_NAME)                           \
  auto VAR_NAME##_out_tensor =                                                 \
      Output(OUTPUT_##OUTPUT_NAME, VAR_NAME##_tensor.sizes(), at::dtype<T>()); \
  VAR_NAME##_out_ = VAR_NAME##_out_tensor->template mutable_data<T>();

    CAFFE2_YF_READ_OUTPUT(PARAM, param)
    CAFFE2_YF_READ_OUTPUT(MOMENT, moment)
    CAFFE2_YF_READ_OUTPUT(LR_AVG, lr_avg)
    CAFFE2_YF_READ_OUTPUT(MU_AVG, mu_avg)
    CAFFE2_YF_READ_OUTPUT(CURV_WIN, curv_win)
    CAFFE2_YF_READ_OUTPUT(G_AVG, g_avg)
    CAFFE2_YF_READ_OUTPUT(G2_AVG, g2_avg)
    CAFFE2_YF_READ_OUTPUT(SCALARS_MEMORY, scalars_memory)
#undef CAFFE2_YF_READ_OUTPUT

    T* out_memory_it = scalars_memory_out_ - 1;
    g_norm_avg_out_ = ++out_memory_it;
    g_norm2_avg_out_ = ++out_memory_it;
    g_norm2_min_avg_out_ = ++out_memory_it;
    g_norm2_max_avg_out_ = ++out_memory_it;
    distance_avg_out_ = ++out_memory_it;

#define CAFFE2_YF_INIT_VECTOR(NAME) \
    ReinitializeTensor(&NAME##_tensor_, {D_}, at::dtype<T>().device(Context::GetDeviceType())); \
    NAME##_ = NAME##_tensor_.template mutable_data<T>();

    CAFFE2_YF_INIT_VECTOR(aux_vector)
    CAFFE2_YF_INIT_VECTOR(g_deb)
    CAFFE2_YF_INIT_VECTOR(g2_deb)
    CAFFE2_YF_INIT_VECTOR(g_deb2)
#undef CAFFE2_YF_INIT_VECTOR

#define CAFFE2_YF_INIT_SCALAR(NAME) \
      ReinitializeTensor(&NAME##_tensor_, {1}, at::dtype<T>().device(Context::GetDeviceType())); \
      NAME##_ = NAME##_tensor_.template mutable_data<T>();

    CAFFE2_YF_INIT_SCALAR(aux_scalar)
    CAFFE2_YF_INIT_SCALAR(distance)
    CAFFE2_YF_INIT_SCALAR(distance_deb)
    CAFFE2_YF_INIT_SCALAR(g_norm)
    CAFFE2_YF_INIT_SCALAR(g_norm_deb)
    CAFFE2_YF_INIT_SCALAR(g_norm2)
    CAFFE2_YF_INIT_SCALAR(g_norm2_max)
    CAFFE2_YF_INIT_SCALAR(g_norm2_max_deb)
    CAFFE2_YF_INIT_SCALAR(g_norm2_min)
    CAFFE2_YF_INIT_SCALAR(g_norm2_min_deb)
    CAFFE2_YF_INIT_SCALAR(g_norm2_deb)
    CAFFE2_YF_INIT_SCALAR(lr)
    CAFFE2_YF_INIT_SCALAR(lr_deb)
    CAFFE2_YF_INIT_SCALAR(mu_deb)
    CAFFE2_YF_INIT_SCALAR(mu)
    CAFFE2_YF_INIT_SCALAR(variance)
#undef CAFFE2_YF_INIT_SCALAR

    debias_factor_ = ZeroDebiasFactor();
    MomentumSgdUpdate();
    AfterApply();
    return true;
  }

 protected:
  int curv_win_width_;
  bool nesterov_;
  bool zero_debias_;

  T epsilon_;
  T beta_;
  T debias_factor_;

  int D_;

// Temporary memory on device, listed all variables used in calculations
#define CAFFE2_YF_DEFINE_TENSOR(NAME) \
  Tensor NAME##_tensor_;              \
  T* NAME##_;

  CAFFE2_YF_DEFINE_TENSOR(aux_vector)
  CAFFE2_YF_DEFINE_TENSOR(g_deb)
  CAFFE2_YF_DEFINE_TENSOR(g2_deb)
  CAFFE2_YF_DEFINE_TENSOR(g_deb2)

  CAFFE2_YF_DEFINE_TENSOR(aux_scalar)
  CAFFE2_YF_DEFINE_TENSOR(distance)
  CAFFE2_YF_DEFINE_TENSOR(distance_deb)
  CAFFE2_YF_DEFINE_TENSOR(g_norm)
  CAFFE2_YF_DEFINE_TENSOR(g_norm_deb)
  CAFFE2_YF_DEFINE_TENSOR(g_norm2)
  CAFFE2_YF_DEFINE_TENSOR(g_norm2_deb)
  CAFFE2_YF_DEFINE_TENSOR(g_norm2_max)
  CAFFE2_YF_DEFINE_TENSOR(g_norm2_max_deb)
  CAFFE2_YF_DEFINE_TENSOR(g_norm2_min)
  CAFFE2_YF_DEFINE_TENSOR(g_norm2_min_deb)
  CAFFE2_YF_DEFINE_TENSOR(lr)
  CAFFE2_YF_DEFINE_TENSOR(lr_deb)
  CAFFE2_YF_DEFINE_TENSOR(mu)
  CAFFE2_YF_DEFINE_TENSOR(mu_deb)
  CAFFE2_YF_DEFINE_TENSOR(variance)

  Tensor scratch_tensor_{Context::GetDeviceType()};

#undef CAFFE2_YF_DEFINE_TENSOR

  // Input tensors' data
  const T* param_;
  const T* moment_;
  const T* lr_avg_;
  const T* mu_avg_;
  const T* curv_win_;
  const T* g_avg_;
  const T* g2_avg_;
  const T* scalars_memory_;
  const T* grad_;
  int iter_;

  // Scalar data from scalars_memory_ input tensor
  const T* g_norm_avg_;
  const T* g_norm2_avg_;
  const T* g_norm2_min_avg_;
  const T* g_norm2_max_avg_;
  const T* distance_avg_;

  // Output tensors' data

  T* param_out_;
  T* moment_out_;
  T* lr_avg_out_;
  T* mu_avg_out_;
  T* curv_win_out_;
  T* g_avg_out_;
  T* g2_avg_out_;
  T* scalars_memory_out_;

  // Scalar data from scalars_memory_ output tensor
  T* g_norm_avg_out_;
  T* g_norm2_avg_out_;
  T* g_norm2_min_avg_out_;
  T* g_norm2_max_avg_out_;
  T* distance_avg_out_;

  INPUT_TAGS(
      PARAM,
      MOMENT,
      LR_AVG,
      MU_AVG,
      CURV_WIN,
      G_AVG,
      G2_AVG,
      SCALARS_MEMORY,
      GRAD,
      ITER);
  OUTPUT_TAGS(
      OUTPUT_PARAM,
      OUTPUT_MOMENT,
      OUTPUT_LR_AVG,
      OUTPUT_MU_AVG,
      OUTPUT_CURV_WIN,
      OUTPUT_G_AVG,
      OUTPUT_G2_AVG,
      OUTPUT_SCALARS_MEMORY);
};

} // namespace caffe2

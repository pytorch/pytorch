#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

// Adam
template <typename Context>
void adam_update(
    int N,
    const float* g,
    const float* m,
    const float* v,
    float* ng,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr,
    Context* /*context*/) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    ng[i] = lr[0] * correction * mi / (std::sqrt(vi) + eps_hat);
  }
}

template <typename Context>
void adam_compute(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr,
    Context* /*context*/) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    nw[i] = w[i] + lr[0] * correction * mi / (std::sqrt(vi) + eps_hat);
  }
}

template <typename Context>
void adam_compute_smart_decay(
    int N,
    long int t,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    const int64_t* lastSeenIn,
    float* nw,
    float* nm,
    float* nv,
    int64_t* lastSeenOut,
    float beta1,
    float beta2,
    float eps_hat,
    //float correction,
    const float* lr,
    Context* /*context*/) {
  float k = (float)(t - lastSeenIn[0]);
  lastSeenOut[0] = t;
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    // The number of steps since this param was last seen.
    // We don't need integer precision for k.  Float is fine and it's faster to convert here.
    // Same as sparse Adam except v is decayed by beta2^k rather than beta2
    // Catchup = \sum_{i=1}^{k-1}\beta_1^i = \beta_1 \left(\frac{1-\beta_1^k}{1-\beta_1}\right)
    float catchup = 0.0;
    if (k > 1) {
        catchup = m[i] * beta1 * (1 - powf(beta1, k-1)) / (1 - beta1);
    }
    float mi = nm[i] = m[i] * powf(beta1, k) + gi * (1 - beta1);
    float vi = nv[i] = v[i] * powf(beta2, k) + gi * gi * (1 - beta2);
    nw[i] = w[i] + (lr[0] * (mi + catchup)) / (std::sqrt(vi) + eps_hat);
  }
}

template <typename Context>
void adam_compute_output_grad(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float* ng,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr,
    Context* /*context*/) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    float ngi = ng[i] = correction * mi / (std::sqrt(vi) + eps_hat);
    nw[i] = w[i] + lr[0] * ngi;
  }
}

// RAdam
template <typename Context>
void radam_update(
    int N,
    const float* g,
    const float* m,
    const float* v,
    float* ng,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float beta1_correction,
    float correction,
    float rho_t,
    float r_correction,
    const float* lr,
    Context* /*context*/) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);

    if (rho_t >= 5.) {
      float r_t =
          std::sqrt(((rho_t - 4.) * (rho_t - 2.)) / rho_t) * r_correction;
      ng[i] = lr[0] * r_t * correction * mi / (std::sqrt(vi) + eps_hat);
    } else {
      ng[i] = lr[0] * beta1_correction * mi;
    }
  }
}

template <typename Context>
void radam_compute(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float beta1_correction,
    float correction,
    float rho_t,
    float r_correction,
    const float* lr,
    Context* /*context*/) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);

    if (rho_t >= 5.) {
      float r_t =
          std::sqrt(((rho_t - 4.) * (rho_t - 2.)) / rho_t) * r_correction;
      nw[i] = w[i] + lr[0] * r_t * correction * mi / (std::sqrt(vi) + eps_hat);
    } else {
      nw[i] = w[i] + lr[0] * beta1_correction * mi;
    }
  }
}

template <typename Context>
void radam_compute_output_grad(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float* ng,
    float beta1,
    float beta2,
    float eps_hat,
    float beta1_correction,
    float correction,
    float rho_t,
    float r_correction,
    const float* lr,
    Context* /*context*/) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    float ngi;

    if (rho_t >= 5.) {
      float r_t =
          std::sqrt(((rho_t - 4.) * (rho_t - 2.)) / rho_t) * r_correction;
      ngi = ng[i] = r_t * correction * mi / (std::sqrt(vi) + eps_hat);
    } else {
      ngi = ng[i] = beta1_correction * mi;
    }
    nw[i] = w[i] + lr[0] * ngi;
  }
}

template <typename T, class Context>
class AdamOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AdamOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        beta1_(this->template GetSingleArgument<float>("beta1", 0.9f)),
        beta2_(this->template GetSingleArgument<float>("beta2", 0.999f)),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)) {}
  bool RunOnDevice() override {
    // Iter live on the CPU
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(ITER, CPU));
    CAFFE_ENFORCE(Input(LR).numel() == 1);
    CAFFE_ENFORCE(Input(GRAD).numel() == Input(PARAM).numel());
    CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENT_1).numel());
    CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENT_2).numel());
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
    Output(OUTPUT_MOMENT_2)->ResizeLike(Input(MOMENT_2));

    const auto iter =
        OperatorBase::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];

    const auto t = iter + 1;
    const auto correction =
        std::sqrt(T(1.) - std::pow(beta2_, t)) / (T(1.) - std::pow(beta1_, t));
    if (OutputSize() == 3) {
      adam_compute<Context>(
          Input(GRAD).numel(),
          Input(PARAM).template data<T>(),
          Input(GRAD).template data<T>(),
          Input(MOMENT_1).template data<T>(),
          Input(MOMENT_2).template data<T>(),
          Output(OUTPUT_PARAM)->template mutable_data<T>(),
          Output(OUTPUT_MOMENT_1)->template mutable_data<T>(),
          Output(OUTPUT_MOMENT_2)->template mutable_data<T>(),
          beta1_,
          beta2_,
          epsilon_,
          correction,
          Input(LR).template data<T>(),
          &context_);
    } else {
      Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
      adam_compute_output_grad<Context>(
          Input(GRAD).numel(),
          Input(PARAM).template data<T>(),
          Input(GRAD).template data<T>(),
          Input(MOMENT_1).template data<T>(),
          Input(MOMENT_2).template data<T>(),
          Output(OUTPUT_PARAM)->template mutable_data<T>(),
          Output(OUTPUT_MOMENT_1)->template mutable_data<T>(),
          Output(OUTPUT_MOMENT_2)->template mutable_data<T>(),
          Output(OUTPUT_GRAD)->template mutable_data<T>(),
          beta1_,
          beta2_,
          epsilon_,
          correction,
          Input(LR).template data<T>(),
          &context_);
    }

    return true;
  }

 protected:
  T beta1_{0.9};
  T beta2_{0.999};
  T epsilon_{1e-8};
  INPUT_TAGS(PARAM, MOMENT_1, MOMENT_2, GRAD, LR, ITER);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, OUTPUT_MOMENT_2, OUTPUT_GRAD);
};

template <typename T, class Context>
class SparseAdamOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseAdamOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        beta1_(this->template GetSingleArgument<float>("beta1", 0.9f)),
        beta2_(this->template GetSingleArgument<float>("beta2", 0.999f)),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
        enableRAdam_(
            this->template GetSingleArgument<bool>("enableRAdam", false)) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_1).numel());
    CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_2).numel());
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).dim()));
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    const auto iter =
        OperatorBase::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];

    const auto t = iter + 1;
    const auto beta1_correction = T(1.) / (T(1.) - std::pow(beta1_, t));
    const auto beta2_correction =
        T(1.) / std::sqrt(T(1.) - std::pow(beta2_, t));
    const auto correction = beta1_correction / beta2_correction;
    const auto rho_inf = T(2.) / (T(1.) - beta2_) - T(1.);
    const auto rho_t = rho_inf -
        T(2.) * t * std::pow(beta2_, t) / (T(1.) - std::pow(beta2_, t));
    const T r_correction = enableRAdam_
        ? std::sqrt(rho_inf / ((rho_inf - T(4.)) * (rho_inf - T(2.))))
        : 0;

    auto block_size = Input(PARAM).numel() / Input(PARAM).size(0);
    auto n = Input(GRAD).numel() / block_size;

    const auto* paramIn = Input(PARAM).template data<T>();
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* moment1In = Input(MOMENT_1).template data<T>();
    const auto* moment2In = Input(MOMENT_2).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* moment1Out = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
    auto* moment2Out = Output(OUTPUT_MOMENT_2)->template mutable_data<T>();

    if (OutputSize() == 3) {
      for (auto i = 0; i < n; ++i) {
        auto idx = indices[i];

        if (block_size == 1) {
          float gi = gradIn[i];
          float mi = moment1Out[idx] =
              moment1In[idx] * beta1_ + gi * (1 - beta1_);
          float vi = moment2Out[idx] =
              moment2In[idx] * beta2_ + gi * gi * (1 - beta2_);

          if (!enableRAdam_) {
            paramOut[idx] = paramIn[idx] +
                lr[0] * correction * mi / (std::sqrt(vi) + epsilon_);
          } else {
            // the SMA condition follows author's implementation
            // 5 is more conservative since it's an approximated value
            if (rho_t >= T(5.)) {
              float r_t =
                  std::sqrt(((rho_t - T(4.)) * (rho_t - T(2.))) / rho_t) *
                  r_correction;
              // epsilon_ is not included in paper, but it is added in author's
              // implementation:
              // https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py#L85
              paramOut[idx] = paramIn[idx] +
                  lr[0] * r_t * correction * mi / (std::sqrt(vi) + epsilon_);
            } else {
              paramOut[idx] = paramIn[idx] + lr[0] * beta1_correction * mi;
            }
          }
        } else {
          auto offsetI = i * block_size;
          auto offsetIdx = idx * block_size;

#ifndef NDEBUG
          CAFFE_ENFORCE_GE(
              Input(PARAM).numel(),
              block_size + offsetIdx,
              this->debug_def().input(PARAM),
              ", out of bound,  idx:",
              idx,
              " for input i:",
              i,
              " and block size:",
              block_size);
          CAFFE_ENFORCE_GE(
              Input(GRAD).numel(),
              block_size + offsetI,
              this->debug_def().input(GRAD),
              ", out of bound idx, idx:",
              idx,
              " for input i:",
              i);
#endif
          if (!enableRAdam_) {
            adam_compute(
                block_size,
                paramIn + offsetIdx,
                gradIn + offsetI,
                moment1In + offsetIdx,
                moment2In + offsetIdx,
                paramOut + offsetIdx,
                moment1Out + offsetIdx,
                moment2Out + offsetIdx,
                beta1_,
                beta2_,
                epsilon_,
                correction,
                lr,
                &context_);
          } else {
            radam_compute(
                block_size,
                paramIn + offsetIdx,
                gradIn + offsetI,
                moment1In + offsetIdx,
                moment2In + offsetIdx,
                paramOut + offsetIdx,
                moment1Out + offsetIdx,
                moment2Out + offsetIdx,
                beta1_,
                beta2_,
                epsilon_,
                beta1_correction,
                correction,
                rho_t,
                r_correction,
                lr,
                &context_);
          }
        }
      }
    } else {
      Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
      auto* gradOut = Output(OUTPUT_GRAD)->template mutable_data<T>();
      for (auto i = 0; i < n; ++i) {
        auto idx = indices[i];

        if (block_size == 1) {
          float gi = gradIn[i];
          float mi = moment1Out[idx] =
              moment1In[idx] * beta1_ + gi * (1 - beta1_);
          float vi = moment2Out[idx] =
              moment2In[idx] * beta2_ + gi * gi * (1 - beta2_);
          float ngi;

          if (!enableRAdam_) {
            ngi = gradOut[i] = correction * mi / (std::sqrt(vi) + epsilon_);
          } else {
            if (rho_t >= T(5.)) {
              float r_t =
                  std::sqrt(((rho_t - T(4.)) * (rho_t - T(2.))) / rho_t) *
                  r_correction;
              ngi = gradOut[i] =
                  r_t * correction * mi / (std::sqrt(vi) + epsilon_);
            } else {
              ngi = gradOut[i] = beta1_correction * mi;
            }
          }

          paramOut[idx] = paramIn[idx] + lr[0] * ngi;
        } else {
          auto offsetI = i * block_size;
          auto offsetIdx = idx * block_size;

#ifndef NDEBUG
          CAFFE_ENFORCE_GE(
              Input(PARAM).numel(),
              block_size + offsetIdx,
              this->debug_def().input(PARAM),
              ", out of bound,  idx:",
              idx,
              " for input i:",
              i,
              " and block size:",
              block_size);
          CAFFE_ENFORCE_GE(
              Input(GRAD).numel(),
              block_size + offsetI,
              this->debug_def().input(GRAD),
              ", out of bound idx, idx:",
              idx,
              " for input i:",
              i);
#endif
          if (!enableRAdam_) {
            adam_compute_output_grad(
                block_size,
                paramIn + offsetIdx,
                gradIn + offsetI,
                moment1In + offsetIdx,
                moment2In + offsetIdx,
                paramOut + offsetIdx,
                moment1Out + offsetIdx,
                moment2Out + offsetIdx,
                gradOut + offsetI,
                beta1_,
                beta2_,
                epsilon_,
                correction,
                lr,
                &context_);
          } else {
            radam_compute_output_grad(
                block_size,
                paramIn + offsetIdx,
                gradIn + offsetI,
                moment1In + offsetIdx,
                moment2In + offsetIdx,
                paramOut + offsetIdx,
                moment1Out + offsetIdx,
                moment2Out + offsetIdx,
                gradOut + offsetI,
                beta1_,
                beta2_,
                epsilon_,
                beta1_correction,
                correction,
                rho_t,
                r_correction,
                lr,
                &context_);
          }
        }
      }
    }
    return true;
  }

 protected:
  T beta1_;
  T beta2_;
  T epsilon_;
  T enableRAdam_;
  INPUT_TAGS(PARAM, MOMENT_1, MOMENT_2, INDICES, GRAD, LR, ITER);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, OUTPUT_MOMENT_2, OUTPUT_GRAD);
};

template <typename T, class Context>
class SmartDecaySparseAdamOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SmartDecaySparseAdamOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        beta1_(this->template GetSingleArgument<float>("beta1", 0.9f)),
        beta2_(this->template GetSingleArgument<float>("beta2", 0.999f)),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_1).numel());
    CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_2).numel());
    CAFFE_ENFORCE_EQ(Input(PARAM).size(0), Input(LAST_SEEN).numel());
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).dim()));
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    const auto iter =
        OperatorBase::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];

    const int64_t t = iter + 1;

    auto block_size = Input(PARAM).numel() / Input(PARAM).size(0);
    auto n = Input(GRAD).numel() / block_size;

    const auto* paramIn = Input(PARAM).template data<T>();
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* moment1In = Input(MOMENT_1).template data<T>();
    const auto* moment2In = Input(MOMENT_2).template data<T>();
    const int64_t* lastSeenIn = Input(LAST_SEEN).template data<int64_t>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* moment1Out = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
    auto* moment2Out = Output(OUTPUT_MOMENT_2)->template mutable_data<T>();
    int64_t* lastSeenOut = Output(OUTPUT_LAST_SEEN)->template mutable_data<int64_t>();

    for (auto i = 0; i < n; ++i) {
        auto idx = indices[i];
        auto offsetI = i * block_size;
        auto offsetIdx = idx * block_size;
        adam_compute_smart_decay(
            block_size,
            t,
            paramIn + offsetIdx,
            gradIn + offsetI,
            moment1In + offsetIdx,
            moment2In + offsetIdx,
            lastSeenIn + idx,
            paramOut + offsetIdx,
            moment1Out + offsetIdx,
            moment2Out + offsetIdx,
            lastSeenOut + idx,
            beta1_,
            beta2_,
            epsilon_,
            lr,
            &context_);
    }

    return true;
  }

 protected:
  T beta1_;
  T beta2_;
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, MOMENT_2, LAST_SEEN, INDICES, GRAD, LR, ITER);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, OUTPUT_MOMENT_2, OUTPUT_LAST_SEEN);
};

template <typename T, class Context>
class RowWiseSparseAdamOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RowWiseSparseAdamOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        beta1_(this->template GetSingleArgument<float>("beta1", 0.9f)),
        beta2_(this->template GetSingleArgument<float>("beta2", 0.999f)),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_1).numel());
    CAFFE_ENFORCE_EQ(Input(PARAM).sizes()[0], Input(MOMENT_2).numel());
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).dim()));
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    const auto iter =
        OperatorBase::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];

    const auto t = iter + 1;
    const auto correction =
        std::sqrt(T(1.) - std::pow(beta2_, t)) / (T(1.) - std::pow(beta1_, t));

    auto block_size = Input(PARAM).numel() / Input(PARAM).size(0);
    auto n = Input(GRAD).numel() / block_size;

    const auto* paramIn = Input(PARAM).template data<T>();
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* moment1In = Input(MOMENT_1).template data<T>();
    const auto* moment2In = Input(MOMENT_2).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* moment1Out = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
    auto* moment2Out = Output(OUTPUT_MOMENT_2)->template mutable_data<T>();

    if (OutputSize() == 3) {
      for (auto i = 0; i < n; ++i) {
        auto idx = indices[i];

        if (block_size == 1) {
          float gi = gradIn[i];
          float mi = moment1Out[idx] =
              moment1In[idx] * beta1_ + gi * (1 - beta1_);
          float vi = moment2Out[idx] =
              moment2In[idx] * beta2_ + gi * gi * (1 - beta2_);
          paramOut[idx] = paramIn[idx] +
              lr[0] * correction * mi / (std::sqrt(vi) + epsilon_);

        } else {
          auto offsetI = i * block_size;
          auto offsetIdx = idx * block_size;

#ifndef NDEBUG
          CAFFE_ENFORCE_GE(
              Input(PARAM).numel(),
              block_size + offsetIdx,
              this->debug_def().input(PARAM),
              ", out of bound,  idx:",
              idx,
              " for input i:",
              i,
              " and block size:",
              block_size);
          CAFFE_ENFORCE_GE(
              Input(GRAD).numel(),
              block_size + offsetI,
              this->debug_def().input(GRAD),
              ", out of bound idx, idx:",
              idx,
              " for input i:",
              i);
#endif

          const float* w = paramIn + offsetIdx;
          const float* g = gradIn + offsetI;
          const float* m1 = moment1In + offsetIdx;
          const float* m2 = moment2In + idx;
          float* nw = paramOut + offsetIdx;
          float* nm1 = moment1Out + offsetIdx;
          float* nm2 = moment2Out + idx;

          float m2_sum = 0.;
          for (auto j = 0; j < block_size; ++j) {
            float gj = g[j];
            m2_sum += gj * gj;
          }
          float vi = nm2[0] =
              m2[0] * beta2_ + (m2_sum / block_size) * (1 - beta2_);
          for (auto j = 0; j < block_size; ++j) {
            float mi = nm1[j] = m1[j] * beta1_ + g[j] * (1 - beta1_);
            nw[j] = w[j] + lr[0] * correction * mi / (std::sqrt(vi) + epsilon_);
          }
        }
      }
    } else {
      Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
      auto* gradOut = Output(OUTPUT_GRAD)->template mutable_data<T>();
      for (auto i = 0; i < n; ++i) {
        auto idx = indices[i];

        if (block_size == 1) {
          float gi = gradIn[i];
          float mi = moment1Out[idx] =
              moment1In[idx] * beta1_ + gi * (1 - beta1_);
          float vi = moment2Out[idx] =
              moment2In[idx] * beta2_ + gi * gi * (1 - beta2_);
          float ngi = gradOut[i] = correction * mi / (std::sqrt(vi) + epsilon_);
          paramOut[idx] = paramIn[idx] + lr[0] * ngi;

        } else {
          auto offsetI = i * block_size;
          auto offsetIdx = idx * block_size;

#ifndef NDEBUG
          CAFFE_ENFORCE_GE(
              Input(PARAM).numel(),
              block_size + offsetIdx,
              this->debug_def().input(PARAM),
              ", out of bound,  idx:",
              idx,
              " for input i:",
              i,
              " and block size:",
              block_size);
          CAFFE_ENFORCE_GE(
              Input(GRAD).numel(),
              block_size + offsetI,
              this->debug_def().input(GRAD),
              ", out of bound idx, idx:",
              idx,
              " for input i:",
              i);
#endif

          const float* w = paramIn + offsetIdx;
          const float* g = gradIn + offsetI;
          const float* m1 = moment1In + offsetIdx;
          const float* m2 = moment2In + idx;
          float* nw = paramOut + offsetIdx;
          float* nm1 = moment1Out + offsetIdx;
          float* nm2 = moment2Out + idx;
          float* ng = gradOut + offsetI;

          float m2_sum = 0.;
          for (auto j = 0; j < block_size; ++j) {
            float gj = g[j];
            m2_sum += gj * gj;
          }
          float vi = nm2[0] =
              m2[0] * beta2_ + (m2_sum / block_size) * (1 - beta2_);
          for (auto j = 0; j < block_size; ++j) {
            float mi = nm1[j] = m1[j] * beta1_ + g[j] * (1 - beta1_);
            float ngi = ng[j] = correction * mi / (std::sqrt(vi) + epsilon_);
            nw[j] = w[j] + lr[0] * ngi;
          }
        }
      }
    }
    return true;
  }

 protected:
  T beta1_;
  T beta2_;
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, MOMENT_2, INDICES, GRAD, LR, ITER);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, OUTPUT_MOMENT_2, OUTPUT_GRAD);
};

} // namespace caffe2
